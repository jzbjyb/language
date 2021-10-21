#!/usr/bin/env bash

export PYTHONPATH="../../":"${PYTHONPATH}"

model=gs://realm-data/cc_news_pretrained/embedder
example=data/biology_intro_500k/examples.tfr
encoded=data/biology_intro_500k/encoded/encoded.ckpt

python -m predict.encode_blocks \
  --retriever_module_path=${model} \
  --examples_path=${example} \
  --encoded_path=${encoded} \
  --num_blocks 898840 \
  --num_threads 16 \
  --batch_size 32
