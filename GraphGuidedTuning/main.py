#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
import json

import numpy as np
import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
from util_data import load_data
# from util_data import DataCollatorForSeq2Seq as _DataCollatorForSeq2Seq
from transformers import DataCollatorForSeq2Seq
# from util_data import Seq2SeqTrainer
from transformers import Seq2SeqTrainer
from peft import LoraConfig, get_peft_model, PeftModel
import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
    get_scheduler
)
import torch.optim as optim
# from GraphGuideCotTuning.arguments import ModelArguments, DataTrainingArguments
from arguments import ModelArguments, DataTrainingArguments

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    # datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load dataset
    train_data, eval_data, name_maps, graph_feature, entities, entity_embeddings = load_data(data_args)

    graph_size = graph_feature.shape[-1]
    entity_size = entity_embeddings.shape[-1]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    if model_args.graph_guide and model_args.embedding_enhance:
        from modelT5 import T5ForGraphGuideGeneration as Model
        model = Model.from_pretrained(model_args.model_name_or_path, graph_size=graph_size, entity_size=entity_size)

    else:
        from transformers import T5ForConditionalGeneration as Model

        model = Model.from_pretrained(model_args.model_name_or_path)


    if training_args.resume_from_checkpoint is not None:

        model = PeftModel.from_pretrained(model, training_args.resume_from_checkpoint, is_trainable=True)
    else:
        if isinstance(model_args.target_modules,str):
            target_modules = model_args.target_modules.split(",")
        else:
            target_modules = model_args.target_modules
        config = LoraConfig(
            r=model_args.r,
            lora_alpha=model_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=model_args.lora_dropout,
            bias=model_args.bias,
            task_type=model_args.task_type
        )

        model = get_peft_model(model, config)

    model.print_trainable_parameters()
    print(model)

    # Preprocessing the datasets.
    if model_args.graph_guide:
        from util_data import ProcessPlanningDatasetGraphForT5 as ProcessPlanningDataset

        train_dataset = ProcessPlanningDataset(
            train_data,
            name_maps,
            tokenizer,
            data_args.max_source_length,
            data_args.max_target_length,
            graph_feature,
            entities,
            entity_embeddings,
            args=data_args,
            max_sample=data_args.max_train_samples
        )
        eval_dataset = ProcessPlanningDataset(
            eval_data,
            name_maps,
            tokenizer,
            data_args.max_source_length,
            data_args.max_target_length,
            graph_feature,
            entities,
            entity_embeddings,
            args=data_args,
            max_sample=data_args.max_eval_samples
        )
        predict_dataset = eval_dataset

    else:
        from util_data import ProcessPlanningDatasetForT5 as ProcessPlanningDataset

        train_dataset = ProcessPlanningDataset(
            train_data,
            tokenizer,
            data_args.max_source_length,
            data_args.max_target_length,
            args=data_args,
            max_sample=data_args.max_train_samples
        )
        eval_dataset = ProcessPlanningDataset(
            eval_data,
            tokenizer,
            data_args.max_source_length,
            data_args.max_target_length,
            args=data_args,
            max_sample=data_args.max_eval_samples
        )
        predict_dataset = eval_dataset


    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=False
    )

    # Metric
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # 过滤不合理的预测
        valid_range = tokenizer.vocab_size
        preds = np.clip(preds, 0, valid_range - 1)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:

            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        score_dict = {
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "bleu-4": [] ,
        }
        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            rouge = Rouge()
            scores = rouge.get_scores(' '.join(hypothesis) , ' '.join(reference))
            result = scores[0]
            
            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )

    # Initialize optimizer before trainer
    optimizer = optim.AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)

    # Learning Rate Scheduler
    num_training_steps = training_args.max_steps
    lr_scheduler = get_scheduler(
        name="cosine",  # Use linear decay to avoid lr = 0
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=num_training_steps
    )
    training_args.use_peft = True
    # Pass optimizer & scheduler to Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer if not model_args.lora else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        optimizers=(optimizer, lr_scheduler)  # Ensure optimizer is set
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        # elif last_checkpoint is not None:
        #     checkpoint = last_checkpoint

        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        # Manual Restoration
        if checkpoint is not None:
            # Restore optimizer state
            trainer.optimizer.load_state_dict(torch.load(f"{checkpoint}/optimizer.pt"))

            # Restore scheduler state
            trainer.lr_scheduler.load_state_dict(torch.load(f"{checkpoint}/scheduler.pt"))

            rng_state = torch.load(f"{checkpoint}/rng_state.pth")

            if isinstance(rng_state, dict) and "cuda" in rng_state:  # Dictionary case
                torch.cuda.set_rng_state(rng_state["cuda"])
            elif isinstance(rng_state, dict) and "cpu" in rng_state:  # Dictionary case
                torch.set_rng_state(rng_state["cpu"])

            with open(f"{checkpoint}/trainer_state.json") as f:
                trainer.state = json.load(f)

            training_args = torch.load(f"{checkpoint}/training_args.bin")

        train_result = trainer.train(resume_from_checkpoint=False)

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        # trainer.save_state()

    # Evaluation
    results = {}
    max_seq_length = data_args.max_source_length + data_args.max_target_length + 1
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval", do_sample=True, top_p=0.7, max_length=max_seq_length, temperature=0.95)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict", max_length=max_seq_length, do_sample=True, top_p=0.7, temperature=0.95)
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        valid_range = tokenizer.vocab_size

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:

                predictions = np.clip(predict_results.predictions, 0, valid_range - 1)

                predictions = tokenizer.batch_decode(
                    predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                labels = tokenizer.batch_decode(
                    predict_results.label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                labels = [label.strip() for label in labels]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    for p, l in zip(predictions, labels):
                        res = json.dumps({"labels": l, "predict": p}, ensure_ascii=False)
                        writer.write(f"{res}\n")
    return results


if __name__ == "__main__":
    main()
