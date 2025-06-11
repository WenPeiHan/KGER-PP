#-*-coding -utf-8 -*-
import sys
import os
from model.utils import update_config, test_db_connection, logging
# from model.export import export
from model.preprocess import preprocess_exported_data
from model.train_embeddings import train_embeddings

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Embedding-related parameter settings')
    parser.add_argument("--project_name", type=str, default="fxq", help="name of the root folder where embeddings are stored")
    parser.add_argument("--train", type=str, default="train", help="train or test")
    parser.add_argument("--model", type=str, default="TransE")
    parser.add_argument("--negative_sample_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--hidden_dim", type=int, default=400)
    parser.add_argument("--gamma", type=float, default=12.0)
    parser.add_argument("--adversarial_temperature", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--config_path", type=str, default="config.yml", help="path to a yml configuration file")
    parser.add_argument("--negative_adversarial_sampling", action='store_true' if not True else 'store_false', default=True)
    parser.add_argument("--double_entity_embedding", action='store_true',)
    parser.add_argument("--double_relation_embedding", action='store_true',)
    parser.add_argument("--init_checkpoint", action='store_true')
    parser.add_argument("--save_checkpoint_steps",default=500)
    parser.add_argument("--output",default="output",type=str)
    return parser.parse_args()


def embed():
    # train,project_name,model,negative_sample_size,batch_size,hidden_dim,gamma,adversarial_temperature,learning_rate,max_steps,
    # config_path,negative_adversarial_sampling,double_entity_embedding):
    """Command line interface for training and generating graph embeddings
    """
    try:
        # test run to check for db connection
        args = parse_args()
        if args.train == "train":
            update_config(
                project_name = args.project_name,
                config_path=args.config_path,
                model = args.model,
                negative_sample_size = args.negative_sample_size,
                batch_size = args.batch_size,
                hidden_dim = args.hidden_dim,
                gamma = args.gamma,
                adversarial_temperature = args.adversarial_temperature,
                learning_rate = args.learning_rate,
                max_steps = args.max_steps,
                negative_adversarial_sampling = args.negative_adversarial_sampling,
                double_entity_embedding = args.double_entity_embedding,
                double_relation_embedding = args.double_relation_embedding,
                train = args.train == "train",
                init_checkpoint = args.init_checkpoint,
                save_checkpoint_steps = args.save_checkpoint_steps
            )
            # export(project_name)
            preprocess_exported_data(args.project_name)
            train_embeddings(args.project_name)
            logging.info("Done....")

    except Exception as e:
        logging.info(f"Error: {e}")
        sys.exit(e)


if __name__ == "__main__":
    embed()
