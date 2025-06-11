#-*-coding -utf-8 -*-
from torch.utils.data import Dataset
import os
import json
import numpy as np
import torch
import re
from sklearn.model_selection import train_test_split
from transformers import DataCollatorForSeq2Seq as _DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer as _Seq2SeqTrainer
from typing import Any, Optional, Union
from torch import nn
from collections import defaultdict
import random



random.seed(42)

def read_json(dir):
    # 打开JSON文件并读取内容
    with open(dir, 'r', encoding='utf-8') as json_file:
        file_content = json_file.read()

    # 解析JSON字符串为字典
    data_dict = json.loads(file_content)
    return data_dict

def load_data(args):

    file_path = args.data_file_path
    graph_feature_path = args.graph_feature_path
    entity_embedding_path = args.entity_embedding_path
    entities_path = args.entities_path

    data = read_json(file_path)

    if args.specify_type:
        data = [item for item in data if item["type"] == args.specify_type]

    specific_key = "questions_template"

    if args.type_balance:
        classified = defaultdict(list)
        for item in data:
            # Check if the key exists in the dictionary
            if specific_key in item:
                classified[item[specific_key]].append(item)

        data_classified = dict(classified)
        data_balance = []

        for key, value_list in data_classified.items():
            value_list = [item for item in value_list if item["rationale_use"] == 1]
            if len(value_list) > args.balance_max:
                value_list = random.sample(value_list, 30)
            if len(value_list) < args.balance_min:
                continue
            data_balance.extend(value_list)

        data = data_balance

    classified_train_test = defaultdict(list)
    for item in data:
        # Check if the key exists in the dictionary
        if specific_key in item:
            classified_train_test[item[specific_key]].append(item)

    data_classified_train_test = dict(classified_train_test)
    train_data, test_data = [], []

    for key_b, value_list_b in data_classified_train_test.items():
        train_data_set, test_data_set = train_test_split(value_list_b, test_size=args.train_test_split, random_state=42)
        train_data.extend(train_data_set)
        test_data.extend(test_data_set)


    entities = read_json(entities_path)
    graph_feature = np.load(graph_feature_path)
    entity_embedding = np.load(entity_embedding_path)

    name_maps = dict()
    for item in data:
        name_maps[item["qid"]] = item["qid"] - 1

    return train_data, test_data, name_maps, graph_feature, entities, entity_embedding

def build_pair(data):

    question = data["question"]
    triple = data["triple_text"]
    if len(triple) > 10:
        triple = triple[:10]
    triple_text = ";".join(triple)

    rationale = data["rationale"]

    prompt = "Question: {}".format(question) + " Triple: {}".format(triple_text)
    target = "Rationale：{}".format(rationale)

    return prompt, target

class ProcessPlanningDatasetGraphForT5(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
            self, data, name_maps, tokenizer, source_len, target_len, graph_features, entities, entity_embeddings, max_sample = None, args = None):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = data
        self.source_len = source_len
        self.summ_len = target_len
        self.target = []
        self.source = []
        self.graph_ids = []

        self.entity_ids = []

        entities_to_id = {v['properties']['name']: k for k, v in entities.items()}

        graph_feature_size = graph_features.shape[-1]

        for index, item in enumerate(self.data):

            prompt, target = build_pair(item)

            self.source.append(prompt)
            self.target.append(target)

            phs = list(item["placeholders_value"].values())
            phs_embeddings = []
            for ph in phs:
                if isinstance(ph,int):
                    continue
                entity_id = entities_to_id[ph]
                phs_embeddings.append(entity_embeddings[int(entity_id)])

            # if len(phs_embeddings) < 2:
            #     phs_m = []
            #     phs_m.append(np.mean(np.stack(phs_embeddings), axis=0))
            #     phs_embeddings = phs_m

            if len(phs_embeddings) == 1:
                phs_zero = np.zeros(phs_embeddings[0].shape[0])
                phs_embeddings.append(phs_zero)

            phs_embeddings = np.array(phs_embeddings)

            self.entity_ids.append(phs_embeddings)

            qid = item["qid"]
            if qid in name_maps:
                i_vectors = graph_features[int(name_maps[qid])]
                self.graph_ids.append(i_vectors)
            else:
                self.graph_ids.append(np.zeros((graph_features.shape[1],graph_features.shape[-1])))

            if index == max_sample:
                break

    def __len__(self):
        """returns the length of dataframe"""
        return len(self.target)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source[index])
        target_text = str(self.target[index])
        graph_ids = self.graph_ids[index]
        entity_ids = self.entity_ids[index]

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze().tolist()

        graph_ids = torch.tensor(graph_ids).squeeze()
        entity_ids = torch.tensor(entity_ids).squeeze()

        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "graph_ids": graph_ids,
            "entity_ids": entity_ids,
            "labels": target_ids,
        }

class ProcessPlanningDatasetForT5(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
            self, data, tokenizer, source_len, target_len,  max_sample = None, args = None):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = data
        self.source_len = source_len
        self.summ_len = target_len
        self.target = []
        self.source = []


        for index, item in enumerate(self.data):

            prompt, target = build_pair(item)

            self.source.append(prompt)
            self.target.append(target)

            if index == max_sample:
                break

    def __len__(self):
        """returns the length of dataframe"""
        return len(self.target)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source[index])
        target_text = str(self.target[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze().tolist()

        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": target_ids,
        }

class DataCollatorForSeq2Seq(_DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        output_ids = (
            [feature['output_ids'] for feature in features]
            if 'output_ids' in features[0].keys()
            else None
        )
        if output_ids is not None:
            max_output_length = max(len(out) for out in output_ids)
            if self.pad_to_multiple_of is not None:
                max_output_length = (
                        (
                                max_output_length + self.pad_to_multiple_of - 1) //
                        self.pad_to_multiple_of * self.pad_to_multiple_of
                )
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (
                        max_output_length - len(feature['output_ids'])
                )
                if isinstance(feature['output_ids'], list):
                    feature['output_ids'] = feature['output_ids'] + remainder
                else:
                    feature['output_ids'] = np.concatenate(
                        [feature['output_ids'], remainder]
                    ).astype(np.int64)
        return super().__call__(features, return_tensors)
