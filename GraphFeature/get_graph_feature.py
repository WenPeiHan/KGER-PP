#-*-coding -utf-8 -*-
import os
import json
import argparse
from data_process import merge
from data_process import kopl_node
from data_process import process as process_data
from graph_node_embedding import main as node_embedding
from graph_feature import main as graph_global_feature

def parse_args():
    parser = argparse.ArgumentParser(description='Parameters related to node embeddings.')

    # process data
    parser.add_argument("-op","--one_data_path",default="../DataSet/DatasetKoPL/DataSet_one.json",type=str,help="original data")
    parser.add_argument("-tp","--two_data_path",default="../DataSet/DatasetKoPL/DataSet_two.json",type=str)

    parser.add_argument("--data_folder",default="data",type=str,help="The folder for storing data")
    parser.add_argument("-ap","--all_data_path",default="../DataSet/data/data_20250103.json",type=str,help="The list after merging data of different types")
    parser.add_argument("-np","--node_path",default="node.json",type=str,help="The node data path")
    parser.add_argument("-pd","--processed_data",default="processed_data.json")

    # node embedding
    parser.add_argument("--node_emb_path",default="node_embeddings_{}.npy",type=str)
    parser.add_argument(
        "--add_token",
        action='store_true' if not False else 'store_false',
        default=False,
        help="Whether to add tokens to the large language model. Adding tokens may cause errors. The default value is False.",
    )
    parser.add_argument(
        "--model_path",
        default="D:\Code\PrML\Bert-base-Chinese",
        help="The path of the model used for embedding KoPL nodes",
    )
    # parser.add_argument("--node_target_dim", type=int, default=768, help="Node embedding dimension", )
    parser.add_argument("--model_name", type=str, default="T5_base", help="Node embedding model name", )
    parser.add_argument("--save_dir", default="output", help="The saving path of node embeddings",)
    parser.add_argument("--entities_path", type=str, default="data/entity.txt", help="The domain entity")
    parser.add_argument("-tasp","--tokenizer_add_save_path", default="data/Tokenizer-Add-Token")

    # graph feature
    parser.add_argument("--GAT_head", type=int, default=1)
    parser.add_argument("--GAT_layer", type=int, default=2)
    parser.add_argument("--node_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_negatives", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("-lr","--learning_rate", type=float, default=0.01)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint/checkpoint.pth")


    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    data_folder = args.data_folder
    one_dataset_path = args.one_data_path
    two_dataset_path = args.two_data_path

    # file_path = os.path.join(data_folder,args.all_data_path)
    file_path = os.path.join(args.all_data_path)
    node_path = os.path.join(data_folder,args.node_path)
    processed_data_path = os.path.join(data_folder,args.processed_data)

    if not os.path.exists(file_path):
        all_data = merge(one_dataset_path,two_dataset_path,file_path)
    else:
        print("all data is exist")
        with open(file_path, 'r', encoding='utf-8') as file:
            all_data = json.load(file)

    if not os.path.exists(node_path):
        all_node = kopl_node(all_data,node_path)
    else:
        print("node json is exist")
        with open(node_path, 'r', encoding='utf-8') as file:
            all_node = json.load(file)

    if not os.path.exists(processed_data_path):
        processed_data = process_data(all_node,all_data,processed_data_path)
    else:
        print("processed data is exist")
        with open(processed_data_path,'r', encoding='utf-8') as file:
            processed_data = json.load(file)

    node_emb_path = os.path.join(args.save_dir,args.node_emb_path.format(args.model_name))
    if not os.path.exists(node_emb_path):
        print("Get node embedding")
        node_embedding(args)
    else:
        print("node embedding is exist")

    graph_global_feature(args)