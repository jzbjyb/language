# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
# Lint as: python3
"""ORQA predictions."""
import json
from absl import flags
from absl import logging
from language.orqa.models import orqa_model
import six
import tensorflow.compat.v1 as tf


FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", None, "Model directory.")
flags.DEFINE_string("question_path", None, "Question file path.")
flags.DEFINE_string("answer_path", None, "Answer file path.")
flags.DEFINE_string("predictions_path", None, "Path to file where predictions will be written")
flags.DEFINE_boolean("print_prediction_samples", True,
                     "Whether to print a sample of the predictions.")
flags.DEFINE_string("format", "txt", "Format of the dataset file.")

flags.DEFINE_string('qa_type', 'abstractive',
                    'The type of QA the model performs. Chosen from "extractive", "generative", "multichoice"')


def main(_):
  FLAGS.answer_path = FLAGS.answer_path or FLAGS.question_path
  num_mask_hint = FLAGS.qa_type == 'multichoice'
  params = {k: getattr(FLAGS, k) for k in ['qa_type']}
  predictor = orqa_model.get_predictor(FLAGS.model_dir, params)
  with tf.io.gfile.GFile(FLAGS.question_path) as qfin, \
    tf.io.gfile.GFile(FLAGS.answer_path) as afin, \
    tf.io.gfile.GFile(FLAGS.predictions_path, 'w') as pfout:
    for i, line in enumerate(qfin):
      if FLAGS.format == 'jsonl':
        example = json.loads(line)
        questions = [example['question']]
        answers = ['']
      elif FLAGS.format == 'txt':
        questions = [line.strip()]
        answers = afin.readline().rstrip('\n')
        if FLAGS.qa_type == 'multichoice':
          answers = answers.split('\t')
        else:
          answers = [answers]
        questions = questions * len(answers)
      else:
        raise NotImplementedError
      for j, (question, answer) in enumerate(zip(questions, answers)):
        predictions = predictor(question, answer, num_mask_hint=num_mask_hint)
        pred = six.ensure_text(predictions['answer'], errors='ignore')
        logprob = predictions['logprob']
        pfout.write(('\t' if j > 0 else '') + f'{pred}\t{logprob}')
      pfout.write('\n')
      if FLAGS.print_prediction_samples and i & (i - 1) == 0:
        logging.info(f'[{i}] {question} -> {pred}')

if __name__ == "__main__":
  tf.disable_v2_behavior()
  tf.app.run()
