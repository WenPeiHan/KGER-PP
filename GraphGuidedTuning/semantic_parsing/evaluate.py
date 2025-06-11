#-*-coding -utf-8 -*-
import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import argparse
from langchain.graphs import Neo4jGraph
from tqdm import tqdm
import os
import logging

# --neo4j_url neo4j://localhost:7687 --neo4j_username neo4j --neo4j_password 123456 --data_parsed_path output/semantic_parse_GLM_cypher.json
# 20250219
# --neo4j_url neo4j://localhost:7687 --neo4j_username neo4j --neo4j_password 123456 --data_parsed_path output_change/semantic_parse_GPT_test10_case3_gen2.json


logging.basicConfig(format="%(asctime)s - %(message)s", level=20)

def read_json(dir):
    # 打开JSON文件并读取内容
    with open(dir, 'r', encoding='utf-8') as json_file:
        file_content = json_file.read()

    # 解析JSON字符串为字典
    data_dict = json.loads(file_content)
    return data_dict

def get_value(list_dict):
    dict_value_list = list()
    for index, item in enumerate(list_dict):
        if isinstance(item,dict):
            for value in item.values():
                dict_value_list.append(value)
    return dict_value_list

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the performance of the semantic parsing model of the knowledge graph.')
    parser.add_argument("--neo4j_url",default=None,type=str)
    parser.add_argument("--neo4j_username",default=None,type=str)
    parser.add_argument("--neo4j_password",default=None,type=str)
    parser.add_argument("--data_parsed_path",default=None,type=str)
    # parser.add_argument("--Acc",action="store_true")
    # parser.add_argument("--EM",action="store_true")
    parser.add_argument("--eval_all", action="store_true")
    parser.add_argument("--eval_file",default="",help="The folder for storing the results  ")
    parser.add_argument("--eval_type",default=None,type=int)
    return parser.parse_args()

def predict_answer(args, original_data):
    for item in tqdm(original_data,total=len(original_data),desc="Get answer use Cypher_"):
        cypher_list = item["cypher_"]
        answer_list = []
        error_list = []
        for cypher_ in cypher_list:
            try:
                answer_ = graph.query(cypher_)
                error_list.append(False)
            except Exception as e:
                # logging.info(f"Error: {e}")
                error_list.append(True)
                answer_ = None
            if not answer_ is None:
                while any(isinstance(ans, dict) for ans in answer_):
                    answer_ = get_value(answer_)
            answer_list.append(answer_)

        item["answer_"] = answer_list
        item["error"] = error_list

    output_dir = os.path.dirname(args.data_parsed_path)
    output_name = os.path.splitext(os.path.basename(args.data_parsed_path))[0]
    output = os.path.join(output_dir,output_name + "_cypher.json")
    with open(output, 'w') as file:
        json.dump(original_data, file)
    return original_data

def evaluate_process(args):
    data = read_json(args.data_parsed_path)
    y_true = []
    y_pred = []
    qids = []
    error_list = []
    pred_answer_key = "answer_"
    if not pred_answer_key in data[0].keys():
        data = predict_answer(args, data)

    if not args.eval_type == None:
        data_type = []
        type_template_list = [key for key, value in type_template.items() if value == args.eval_type]
        for item in data:
            if item["questions_template"] in type_template_list:
                data_type.append(item)
        data = data_type

    for item in data:
        if not args.eval_type == None:
            if not type_template[item["questions_template"]] == args.eval_type:
                continue

        qids.append(item["qid"])
        y_true.append(item["answer"])
        y_pred.append(item["answer_"])
        error_list.append(item["error"])

    exact_matches = 0
    total_predictions = len(y_true)
    TP = 0  # True Positives
    FP = 0  # False Positives
    FN = 0  # False Negatives

    if not args.eval_all:
        # As long as there is one correct result, all the results are considered correct.
        for qid, true, preds, error in tqdm(zip(qids, y_true, y_pred, error_list), total=len(y_true)):
            # Convert both lists to sets for order-independent comparison
            TP_, FP_, FN_ = 0, 0, 0
            true_set = set(true)
            for index, pred in enumerate(preds):
                # if index == 2 or index == 1:
                #     continue
                # pred_set = set(pred) if isinstance(pred, list) else set()
                try:
                    pred_set = set(pred) if isinstance(pred, list) else set()
                except:
                    # print(qid)
                    pred_set = None

                # Exact match check: Sets should match exactly (ignoring order)
                if true_set == pred_set:
                    TP_ += 1  # For precision and recall

                elif pred_set:  # False positive (predicted non-empty but incorrect)
                    FP_ += 1

                if true_set:  # False negative (no prediction or incorrect prediction)
                    if pred_set != true_set:
                        FN_ += 1

            # print(TP_, FP_, FN_)
            if TP_ > 0:
                TP += 1
                # FP, FN = 0, 0
            else:
                TP += 0
                try:
                    pred_set = set(preds[-1]) if isinstance(preds[-1], list) else set()
                    if pred_set:  # False positive (predicted non-empty but incorrect)
                        FP += 1
                    if true_set:  # False negative (no prediction or incorrect prediction)
                        if pred_set != true_set:
                            FN += 1
                except Exception:
                    continue
    else:
        # Consider all the generated results as a single separate experiment and accumulate them.
        for qid, true, preds, error in zip(qids, y_true, y_pred, error_list):
            # Convert both lists to sets for order-independent comparison
            true_set = set(true)
            for pred in preds:
                try:
                    pred_set = set(pred) if isinstance(pred, list) else set()
                except:
                    # print(qid)
                    pred_set = None
                # Exact match check: Sets should match exactly (ignoring order)

                if true_set == pred_set:
                    TP += 1  # For precision and recall

                elif pred_set:  # False positive (predicted non-empty but incorrect)
                    FP += 1

                if true_set:  # False negative (no prediction or incorrect prediction)
                    if pred_set != true_set:
                        FN += 1

        total_predictions = len(y_pred[0]) * total_predictions

    exact_matches = TP
    # Precision: Number of correct predictions / Total predictions (non-empty)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0

    # Recall: Number of correct predictions / Total true answers (non-empty)
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0

    # F1 Score: Harmonic mean of precision and recall
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # Exact Match Rate: Proportion of exact matches
    exact_match_rate = exact_matches / total_predictions if total_predictions != 0 else 0

    evaluation_metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "exact_match_rate": exact_match_rate
    }

    # Store the evaluation results
    eval_file = args.eval_file
    if not os.path.exists(eval_file):
        os.makedirs(eval_file)
    eval_file_name = os.path.splitext(os.path.basename(args.data_parsed_path))[0]

    if args.eval_all:
        if args.eval_type == None:
            eval_output = os.path.join(eval_file, eval_file_name + "_eval_all.json")
        else:
            eval_output = os.path.join(eval_file, eval_file_name + "_eval_all_type_{}.json".format(args.eval_type))
    else:
        if args.eval_type == None:
            eval_output = os.path.join(eval_file, eval_file_name + "_eval.json")
        else:
            eval_output = os.path.join(eval_file, eval_file_name + "_eval_type_{}.json".format(args.eval_type))
    with open(eval_output, 'w') as file:
        json.dump(evaluation_metrics, file, indent=4)

    return evaluation_metrics

if __name__ == '__main__':

    args = parse_args()

    graph = Neo4jGraph(
        url=args.neo4j_url,
        username=args.neo4j_username,
        password=args.neo4j_password
    )
    # graph = None

    type_template_path = "../../DataSet/Template/type_template.json"
    type_template = read_json(type_template_path)

    metrics = evaluate_process(args)

    # Display evaluation metrics
    print("Evaluation Metrics:")
    for metric, score in metrics.items():
        print(f"{metric}: {score:.4f}")
