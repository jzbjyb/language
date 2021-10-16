#!/usr/bin/env bash

export PYTHONPATH="../../":"${PYTHONPATH}"

model_dir=trained_models/realm
input_file=~/exp/adapt_knowledge/data/biorel/prompt_qa/test.source  # gs://orqa-data/resplit/WebQuestions.resplit.test.jsonl
output_file=$1

python -m predict.orqa_predict \
  --dataset_path=${input_file} \
  --predictions_path=${output_file} \
  --model_dir=${model_dir}
