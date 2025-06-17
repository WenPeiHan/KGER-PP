# KGER-PP: Knowledge Graph-Enhanced Reasoning for Process Planning based on Large Language Models

The framework consists of three stages: process planning semantic parsing,  structural and semantic enhancement method (StSeEM), and graph reasoning path-guided tuning.

<img src="https://gitee.com/feng-xueqian/pic-go/raw/master/Figure_model_overall_flow.png" alt="Figure_model_overall_flow" style="zoom: 33%;" />
## Instructions

### Process planning semantic parsing

```shell
python semantic_parsing.py \
  --data case/output/data_wo_case_random_100.json \
  --output output_cypher \
  --GPT --temperature 0.3 \
  --case_path case/output/random_case_100.json \
  --case_random \
  --graph_schema case/graph_schema.txt \
  --case_num 5 \
  --generate_num 3
```

### KG Embedding

```shell
python embedding.py \
  --project_name output/TransE_768 \
  --model TransE \
  --hidden_dim 768 \
  --max_steps 2000 \
  --save_checkpoint_steps 500
```

### Graph reasoning feature

```shell
graph_feature.py \
  --model_path /root/autodl-tmp/PrML/flan-t5-base \
  --all_data_path ../DataSet/data/data_en.json \
  --epochs 30 \
  --model_name T5_base \
  --checkpoint_path checkpoint/checkpoint_base.pth
```

### Tuning

```shell
python main.py \
  --do_train \
  --T5 \
  --model_name_or_path /root/autodl-tmp/PrML/flan-t5-base \
  --data_file_path ../DataSet/data/data_en.json \
  --embedding_enhance True \
  --graph_guide True \
  --entity_embedding_path ../MPKG_Embedding/output/TransE_768/entity_embedding.npy \
  --graph_feature_path ../GraphFeature/output/concat_features_T5_base.npy \
  --preprocessing_num_workers 10 \
  --output_dir output/ \
  --overwrite_output_dir \
  --max_source_length 512 \
  --max_target_length 128 \
  --r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules "q,v,k,o,wi_0,wi_1,wo" \
  --per_device_train_batch_size 24 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 16 \
  --predict_with_generate \
  --max_steps 10000 \
  --logging_steps 1000 \
  --save_steps 2000 \
  --type_balance True \
  --learning_rate 3e-4 \
  --weight_decay 0.01 \
  --warmup_steps 1000 \
  --gradient_accumulation_steps 4
```
