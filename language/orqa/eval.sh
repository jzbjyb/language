#!/usr/bin/env bash

export PYTHONPATH="../../":"${PYTHONPATH}"

model_dir=gs://orqa-data/orqa_wq_model
TF_CONFIG='{"cluster": {"chief": ["host:port"]}, "task": {"type": "evaluator", "index": 0}}' \
python -m experiments.orqa_experiment \
  --retriever_module_path=gs://orqa-data/ict \
  --block_records_path=gs://orqa-data/enwiki-20181220/blocks.tfr \
  --data_root=gs://orqa-data/resplit \
  --model_dir=${model_dir} \
  --dataset_name=WebQuestions
