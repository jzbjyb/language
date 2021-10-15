#!/usr/bin/env bash

export PYTHONPATH="../../":"${PYTHONPATH}"

model_dir=$1
input_file=gs://orqa-data/resplit/WebQuestions.resplit.test.jsonl
output_file=$2

python -m predict.orqa_predict \
  --dataset_path=${input_file} \
  --predictions_path=${output_file} \
  --model_dir=${model_dir}
