#-*-coding -utf-8 -*-
import json
import pandas as pd
from .utils import logging
import os
import sys
from pathlib import Path
from .utils import update_config
from .model import KGEModel
import torch
import numpy as np

from torch.utils.data import DataLoader


from .dataloader import TrainDataset
from .dataloader import BidirectionalOneShotIterator

GLOBAL_CONFIG = None
EMBEDDING_SETTINGS = None

DATA_DIRECTORY = None
PROJECT_DIRECTORY = None

class DictAccessor:
    def __init__(self, dictionary):
        self.__dict__ = dictionary

def initialise_config(program):
    from .utils import load_config, load_config_program
    global GLOBAL_CONFIG, EMBEDDING_SETTINGS, DATA_DIRECTORY,PROJECT_DIRECTORY
    GLOBAL_CONFIG = load_config_program(program,"GLOBAL_CONFIG")
    # EMBEDDING_SETTINGS = load_config("EMBEDDING_SETTINGS")
    cwd = os.getcwd()
    PROJECT_DIRECTORY = os.path.join(cwd, GLOBAL_CONFIG["PROJECT_NAME"])
    DATA_DIRECTORY = os.path.join(cwd, GLOBAL_CONFIG["DATA_DIRECTORY"])
    EMBEDDING_SETTINGS = load_config_program(program,"EMBEDDING_SETTINGS")
    EMBEDDING_SETTINGS["model"] = GLOBAL_CONFIG["MODEL"]
    EMBEDDING_SETTINGS = DictAccessor(EMBEDDING_SETTINGS)


def override_config():
    '''
    Override model and data configuration
    '''

    with open(os.path.join(PROJECT_DIRECTORY, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)

    EMBEDDING_SETTINGS.countries = argparse_dict['countries']
    if GLOBAL_CONFIG["DATA_DIRECTORY"] is None:
        GLOBAL_CONFIG["DATA_DIRECTORY"] = argparse_dict['data_path']

    EMBEDDING_SETTINGS.model = argparse_dict['model']
    EMBEDDING_SETTINGS.double_entity_embedding = argparse_dict['double_entity_embedding']
    EMBEDDING_SETTINGS.double_relation_embedding = argparse_dict['double_relation_embedding']
    EMBEDDING_SETTINGS.hidden_dim = argparse_dict['hidden_dim']
    EMBEDDING_SETTINGS.test_batch_size = argparse_dict['test_batch_size']

def config_check():
    train = GLOBAL_CONFIG["TRAIN"]
    if (not train):
        raise ValueError('one of train/val/test mode must be choosed.')
    if EMBEDDING_SETTINGS.init_checkpoint:
        override_config()
    elif DATA_DIRECTORY is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')
    if train and PROJECT_DIRECTORY is None:
        raise ValueError('Where do you want to save your trained model?')
    if PROJECT_DIRECTORY and not os.path.exists(PROJECT_DIRECTORY):
        os.makedirs(PROJECT_DIRECTORY)

def dict_to_identifier(d):
    return "|".join([f"{key}:{value}" for key, value in d.items()])

def get_triple():
    with open(os.path.join(DATA_DIRECTORY, GLOBAL_CONFIG["NODE_DICT_FILE_NAME"]+".dict")) as fin:
        for line in fin:
            node_json = json.loads(line)

    with open(os.path.join(DATA_DIRECTORY, GLOBAL_CONFIG["RELATION_DICT_FILE_NAME"]+".dict")) as fin:
        for line in fin:
            relations_json = json.loads(line)

    nentity = len(node_json)
    nrelation = len(relations_json)
    with open(os.path.join(DATA_DIRECTORY, GLOBAL_CONFIG["JSON_EXPORT_FILE"]+".txt")) as fin:
        triples = []
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((int(h),int(r),int(t)))

    # return entity2id, relation2id,triples
    return nentity,nrelation,triples

def prepare_model():
    kge_model = KGEModel(
        model_name=GLOBAL_CONFIG['MODEL'],
        nentity=EMBEDDING_SETTINGS.nentity,
        nrelation=EMBEDDING_SETTINGS.nrelation,
        hidden_dim=EMBEDDING_SETTINGS.hidden_dim,
        gamma=EMBEDDING_SETTINGS.gamma,
        double_entity_embedding=EMBEDDING_SETTINGS.double_entity_embedding,
        double_relation_embedding=EMBEDDING_SETTINGS.double_relation_embedding
    )

    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if EMBEDDING_SETTINGS.cuda:
        kge_model = kge_model.cuda()
    return kge_model

def prepare_iterator(train_triples):
    train_dataloader_head = DataLoader(
        TrainDataset(train_triples, EMBEDDING_SETTINGS.nentity, EMBEDDING_SETTINGS.nrelation, EMBEDDING_SETTINGS.negative_sample_size, 'head-batch'),
        batch_size=EMBEDDING_SETTINGS.batch_size,
        shuffle=True,
        num_workers=max(1, EMBEDDING_SETTINGS.cpu_num // 2),
        collate_fn=TrainDataset.collate_fn
    )

    train_dataloader_tail = DataLoader(
        TrainDataset(train_triples, EMBEDDING_SETTINGS.nentity, EMBEDDING_SETTINGS.nrelation, EMBEDDING_SETTINGS.negative_sample_size, 'tail-batch'),
        batch_size=EMBEDDING_SETTINGS.batch_size,
        shuffle=True,
        num_workers=max(1, EMBEDDING_SETTINGS.cpu_num // 2),
        collate_fn=TrainDataset.collate_fn
    )

    train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
    return train_iterator


def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    argparse_dict = vars(args)
    with open(os.path.join(PROJECT_DIRECTORY, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(PROJECT_DIRECTORY, 'checkpoint')
    )

    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(PROJECT_DIRECTORY, 'entity_embedding'),
        entity_embedding
    )

    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(PROJECT_DIRECTORY, 'relation_embedding'),
        relation_embedding
    )

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))

def train_embeddings(program):
    try:
        initialise_config(program)
        logging.info(
            "-------------------------PREPARING FOR TRAINING CHECK CONFIG------------------------"
        )
        config_check()
        EMBEDDING_SETTINGS.nentity,EMBEDDING_SETTINGS.nrelation,triples = get_triple()
        train_triples, test_triples = triples[0:1300],triples[1300:-1]

        logging.info(
            "-------------------------Data ready, start loading model------------------------"
        )
        kge_model = prepare_model()
        train_iterator = prepare_iterator(train_triples)
        current_learning_rate = EMBEDDING_SETTINGS.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()),
            lr=EMBEDDING_SETTINGS.learning_rate
        )
        if EMBEDDING_SETTINGS.warm_up_steps:
            warm_up_steps = EMBEDDING_SETTINGS.warm_up_steps
        else:
            warm_up_steps = EMBEDDING_SETTINGS.max_steps // 2

        if EMBEDDING_SETTINGS.init_checkpoint:
            # Restore model from checkpoint directory
            logging.info('Loading checkpoint %s...' % PROJECT_DIRECTORY)
            checkpoint = torch.load(os.path.join(PROJECT_DIRECTORY, "checkpoint"))
            init_step = checkpoint['step']
            kge_model.load_state_dict(checkpoint['model_state_dict'])
            if GLOBAL_CONFIG["TRAIN"]:
                current_learning_rate = checkpoint['current_learning_rate']
                warm_up_steps = checkpoint['warm_up_steps']
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            logging.info('Ramdomly Initializing {} Model...'.format(GLOBAL_CONFIG["MODEL"]))
            init_step = 0

        step = init_step

        logging.info('Start Training...')

        if GLOBAL_CONFIG["TRAIN"]:
            logging.info('learning_rate = %d' % current_learning_rate)

            training_logs = []

            # Training Loop
            for step in range(init_step, EMBEDDING_SETTINGS.max_steps):

                log = kge_model.train_step(kge_model, optimizer, train_iterator, EMBEDDING_SETTINGS)

                training_logs.append(log)

                if step >= warm_up_steps:
                    current_learning_rate = current_learning_rate / 10
                    logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                    optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, kge_model.parameters()),
                        lr=current_learning_rate
                    )
                    warm_up_steps = warm_up_steps * 3

                if step % int(EMBEDDING_SETTINGS.save_checkpoint_steps) == 0:
                    save_variable_list = {
                        'step': step,
                        'current_learning_rate': current_learning_rate,
                        'warm_up_steps': warm_up_steps
                    }
                    save_model(kge_model, optimizer, save_variable_list, EMBEDDING_SETTINGS)

                if step % EMBEDDING_SETTINGS.log_steps == 0:
                    metrics = {}
                    for metric in training_logs[0].keys():
                        metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                    log_metrics('Training average', step, metrics)
                    training_logs = []

                if EMBEDDING_SETTINGS.do_valid and step % EMBEDDING_SETTINGS.valid_steps == 0:
                    logging.info('Evaluating on Valid Dataset...')
                    metrics = kge_model.test_step(kge_model, test_triples, triples, EMBEDDING_SETTINGS)
                    log_metrics('Valid', step, metrics)

            save_variable_list = {
                'step': step,
                'current_learning_rate': current_learning_rate,
                'warm_up_steps': warm_up_steps
            }
            save_model(kge_model, optimizer, save_variable_list, EMBEDDING_SETTINGS)

        logging.info("Done....")
    except Exception as e:
        logging.info("error in training")
        logging.info(e, exc_info=True)
        sys.exit(e)
    return