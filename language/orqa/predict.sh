#!/usr/bin/env bash

export PYTHONPATH="../../":"${PYTHONPATH}"

has_context=false

if [[ "$has_context" == "true" ]]; then
  qa_type=abstractive
  model_dir=trained_models/realm
  output_file=test.pred
  question_file=test.bm25_biology_intro_umls_multiple_context.source.1-9
  answer_file=test.bm25_biology_intro_umls_multiple_context.target.1-9

  python -m predict.orqa_predict \
  --question_path=${question_file} \
  --answer_path=${answer_file} \
  --predictions_path=${output_file} \
  --model_dir=${model_dir} \
  --qa_type=${qa_type} \
  --reader_beam_size 1 \
  --has_context

  exit
fi

qa_type=$1
data_dir=$2  # ~/exp/adapt_knowledge/data/biorel/prompt_mc_333/
model_dir=$3  # trained_models/realm_biology_intro_500k
output_file=$4
question_file=${data_dir}/test.source
answer_file=${data_dir}/test.target

python -m predict.orqa_predict \
  --question_path=${question_file} \
  --answer_path=${answer_file} \
  --predictions_path=${output_file} \
  --model_dir=${model_dir} \
  --qa_type=${qa_type}
