#-*-coding -utf-8 -*-

from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import numpy as np
import os
import json
from tqdm import tqdm
import pickle
import argparse

# set pytorch seed for Liner
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define function to get BERT embeddings
def get_embeddings(text, model, tokenizer, device, seed=42):

    # Fix the random seed to ensure that the results are consistent every time.
    set_seed(seed)

    # Tokenize input text and convert to tensor
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Get embeddings from T5
    with torch.no_grad():
        outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

    # hidden_states = hidden_states.squeeze(0)

    # Pool the embeddings (e.g., mean pooling)
    pooled_embeddings = torch.mean(hidden_states, dim=1).squeeze()

    # Move to CPU and convert to numpy
    return pooled_embeddings.cpu().detach().numpy()

    # return hidden_states.cpu().detach().numpy()

# Save embeddings to a file
def save_embeddings(embeddings, filepath):
    np.save(filepath, embeddings)

# Load embeddings from a file
def load_embeddings(filepath):
    return np.load(filepath)

# load node from json file
def load_node(node_file_path):
    with open(node_file_path, 'r', encoding='utf-8') as file:
        node_list = json.load(file)
    node_text = []
    for index, item in enumerate(node_list):
        formatted_str = ' '.join([f"{key}:{', '.join(map(str, value))}" if isinstance(value, list) else f"{key}:{value}" for key, value in item.items() if key != 'index'])
        node_text.append(formatted_str)
    return node_text

# add special token for BERT
def add_entity_token(args):
    entities_path = args.entities_path

    # load entities list
    entities_list = []
    with open(entities_path, 'r', encoding='utf-8') as file:  # 一般使用utf-8编码，可根据实际调整
        for line in file:
            entities_list.append(line.strip())  # strip()用于去除每行末尾的换行符

    # load tokenizer
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained(args.model_path)

    # add token
    tokenizer.add_tokens(entities_list)
    save_path = args.tasp
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    tokenizer.save_pretrained(save_path)
    return

# Main code
def main(args):
    # Input texts to process
    # node path
    node_path = os.path.join(args.data_folder,args.node_path)
    save_dir = args.save_dir
    model_path = args.model_path
    add_token = args.add_token
    model_name = args.model_name

    node_text_list = load_node(node_path)
    # Paths
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Load pre-trained model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_path).to(device)

    if add_token:
        tokenizer_path = args.tasp
        # add token
        if not os.path.exists(tokenizer_path):
            add_entity_token()
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)


    all_embeddings = []
    # Process and save embeddings for each text
    for idx, text in tqdm(enumerate(node_text_list),desc="Node Embedding",total=len(node_text_list)):
        embeddings = get_embeddings(text, model, tokenizer, device)
        all_embeddings.append(embeddings)


    all_embeddings = np.array(all_embeddings)

    filepath = os.path.join(save_dir, "node_embeddings_{}.npy".format(model_name))
    save_embeddings(all_embeddings, filepath)
