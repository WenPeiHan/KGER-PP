"""Converts graph database exported in jsonl format to tsv format required by PBG
"""
import json
import pandas as pd
from .utils import logging
import os
import sys
from pathlib import Path

GLOBAL_CONFIG = None
json_path = None



def initialise_config(program):
    from .utils import load_config_program

    global GLOBAL_CONFIG
    global json_path
    GLOBAL_CONFIG = load_config_program(program,"GLOBAL_CONFIG")
    json_path = os.path.join(
        os.getcwd(),
        GLOBAL_CONFIG["DATA_DIRECTORY"],
        GLOBAL_CONFIG["JSON_EXPORT_FILE"] + ".json",
    )  # path to the json dump of the graph db


def read_json_file():
    """read exported json(l) file consisting of the graph database
    Returns:
        [list] -- list of dictionaries for each node and relation
    """
    try:
        logging.info(f"READING GRAPH DATA IN JSON FROM {json_path}")
        with open(json_path, "r",encoding="utf8") as json_file:
            json_list = list(json_file)
            json_list = [json.loads(json_string) for json_string in json_list]
        return json_list
    except Exception as e:
        logging.info("error in reading json path")
        logging.info(e, exc_info=True)


def separate_nodes_relations(json_list):
    """Creates two separate dataframes for the nodes and the relationships
    
    Arguments:
        json_list {[list]} -- list of node/relationship dictionaries
    
    Returns:
        [Dataframe] -- dataframes for the nodes and the relationships
    """
    try:
        logging.info(f"SEPARATING NODES AND RELATIONSHIPS")

        graph_df = pd.DataFrame(json_list)
        nodes_df = graph_df[graph_df["type"] == "node"][
            ["id", "type", "labels", "properties"]
        ]
        relation_df = graph_df[graph_df["type"] == "relationship"][
            ["type", "start", "end", "label", "properties"]
        ]
        nodes_df["labels"] = nodes_df["labels"].apply(lambda x: x[0])
        relation_df["start"] = relation_df["start"].apply(lambda x: x["id"])
        relation_df["end"] = relation_df["end"].apply(lambda x: x["id"])
        return nodes_df, relation_df
    except Exception as e:
        logging.info("error in separating nodes and relations tsv")
        logging.info(e, exc_info=True)


def convert_to_tsv(relation_df):
    """Converts the Dataframe to tsv for PBG to read.
    each row is in the triplet format that defines one edge/relationship in the graph
    columns: start,label,end
        - start: id of the 'from' node
        - end: id of the 'to' node
        - label: type of the relationship
    
    Arguments:
        relation_df {[Dataframe]} -- Dataframe in above mentioned format
    """
    try:
        tsv_path = os.path.join(
            os.getcwd(),
            GLOBAL_CONFIG["DATA_DIRECTORY"],
            GLOBAL_CONFIG["TSV_FILE_NAME"] + ".tsv",
        )  # default myproject/data/graph.tsv
        logging.info(f"WRITING TSV FILE TO {tsv_path}")
        relation_df[["start", "label", "end"]].to_csv(
            tsv_path, sep="\t", header=False, index=False
        )
    except Exception as e:
        logging.info("error in converting to tsv")
        logging.info(e, exc_info=True)
        sys.exit(e)

def relation_drop_duplicates(df):
    unique_rows = []
    for index, row in df.iterrows():
        is_duplicate = False
        for unique_row in unique_rows:
            if row['label'] == unique_row['label'] and row['properties'] == unique_row['properties']:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_rows.append(row)
    return pd.DataFrame(unique_rows)

def relation_merge(relation_df, relations):
    merged_data = []
    for index1, row1 in relation_df.iterrows():
        for index2, row2 in relations.iterrows():
            if row1['label'] == row2['label'] and row1['properties'] == row2['properties']:
                merged_row = {**row1, **row2}
                merged_data.append(merged_row)

    return pd.DataFrame(merged_data)

def convert_to_dict(graph_json,node_df,relation_df):
    node_path = os.path.join(
        os.getcwd(),
        GLOBAL_CONFIG["DATA_DIRECTORY"],
        GLOBAL_CONFIG["NODE_DICT_FILE_NAME"] + ".dict",
    )
    relation_path = os.path.join(
        os.getcwd(),
        GLOBAL_CONFIG["DATA_DIRECTORY"],
        GLOBAL_CONFIG["RELATION_DICT_FILE_NAME"] + ".dict",
    )
    geaph_path = os.path.join(
        os.getcwd(),
        GLOBAL_CONFIG["DATA_DIRECTORY"],
        GLOBAL_CONFIG["GRAPH_FILE_NAME"] + ".txt",
    )
    if Path(relation_path).exists() and Path(node_path).exists() and Path(geaph_path).exists():
        logging.info(f"ENTITY FILE {node_path} IS EXIST")
        logging.info(f"RELATION FILE {relation_path} IS EXIST")
        logging.info(f"GRAPH FILE {geaph_path} IS EXIST")
    else:
        logging.info(f"WRITING ENTITY FILE TO {node_path}")
        logging.info(f"WRITING RELATION FILE TO {relation_path}")
        node_dict = node_df[["labels","properties"]].to_dict(orient='index')
        relations = relation_drop_duplicates(relation_df[['label', 'properties']]).reset_index(drop=True)
        relation_dict = relations.to_dict(orient="index")
        relations = relations.reset_index()
        graph = relation_merge(relation_df,relations)[["start","index","end"]]
        with open(node_path, "w") as file:
            json.dump(node_dict,file)
        with open(relation_path,"w") as file:
            json.dump(relation_dict,file)
        graph.to_csv(geaph_path,sep='\t', index=False,header=False)

# entry function
def preprocess_exported_data(program):
    """entry function for converting graph export data in jsonl format to tsv format supported by PBG
    """
    try:
        initialise_config(program)
        logging.info(
            "-------------------------PREPROCESSING DATA------------------------"
        )
        json_list = read_json_file()
        nodes_df, relations_df = separate_nodes_relations(json_list)
        # convert_to_tsv(relations_df)
        convert_to_dict(json_list,nodes_df,relations_df)
        logging.info("Done")
    except Exception as e:
        logging.info("error in preprocessing")
        logging.info(e, exc_info=True)
        sys.exit(e)
