#!/usr/bin/env bash

export PYTHONPATH="../../":"${PYTHONPATH}"

qa_type=$1
data_dir=$2  # ~/exp/adapt_knowledge/data/biorel/prompt_mc_333/
output_file=$3
question_file=${data_dir}/test.source
answer_file=${data_dir}/test.target
model_dir=trained_models/realm

python -m predict.orqa_predict \
  --question_path=${question_file} \
  --answer_path=${answer_file} \
  --predictions_path=${output_file} \
  --model_dir=${model_dir} \
  --qa_type=${qa_type}
