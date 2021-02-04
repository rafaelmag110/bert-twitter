import os
import re
import math
import json

import tensorflow as tf

from official.nlp.bert import configs as bert_configs
from official.nlp.bert import run_classifier
from official.nlp.bert import bert_models
from official.nlp.modeling import models
from official.utils.misc import distribution_utils


BERT_DIR = '/bert_env/bert_models/uncased_L-12_H-768_A-12'
DATA_DIR = '/bert_env/data'
RECORDS_DIR = '/bert_env/records'
MODEL_DIR = '/bert_env/output_dir'

EVAL_BATCH_SIZE = 64
MAX_SEQ_LENGTH = 64

senti_dict = ['negative','neutral','positive']


print('Tensorflow version:', tf.__version__)

def custom_pred_fn(checkpoint_dir):

  LABEL_TYPES_MAP = {'int': tf.int64, 'float': tf.float32}

  with tf.io.gfile.GFile( DATA_DIR+ "/input_meta_data", 'rb') as reader:
    input_meta_data = json.loads(reader.read().decode('utf-8'))
  label_type = LABEL_TYPES_MAP[input_meta_data.get('label_type', 'int')]
  include_sample_weights = input_meta_data.get('has_sample_weights', False)

  bert_config = bert_configs.BertConfig.from_json_file(BERT_DIR +"/bert_config.json")

  strategy = distribution_utils.get_distribution_strategy(   
      distribution_strategy="mirrored",
      num_gpus=1,
      tpu_address=None)
  eval_input_fn = run_classifier.get_dataset_fn(
      RECORDS_DIR + "/eval.tf_record",
      input_meta_data['max_seq_length'],
      EVAL_BATCH_SIZE,
      is_training=False
      # label_type=label_type,
      # include_sample_weights=include_sample_weights
      )
  eval_steps = int( math.ceil(input_meta_data['eval_data_size'] / EVAL_BATCH_SIZE))

  
  with strategy.scope():
    classifier_model = bert_models.classifier_model(
        bert_config, 
        input_meta_data['num_labels'], 
        MAX_SEQ_LENGTH, 
        hub_module_url=None, 
        hub_module_trainable=False)[0]
    checkpoint = tf.train.Checkpoint(model=classifier_model)
    latest_checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    # latest_checkpoint_file = checkpoint_dir + '/bert_model.ckpt'
    assert latest_checkpoint_file
    print('Checkpoint file %s found and restoring from '
                  'checkpoint', latest_checkpoint_file)
    checkpoint.restore(latest_checkpoint_file).assert_existing_objects_matched()
    preds, labels = run_classifier.get_predictions_and_labels(strategy, 
                                                        classifier_model,
                                                         eval_input_fn)
    output_predict_file = os.path.join(checkpoint_dir, 'test_results.tsv')
  with tf.io.gfile.GFile(output_predict_file, 'w') as writer:
    print('***** Predict results *****')
    for probability, label in zip(preds,labels):
      output_line = '\t'.join([senti_dict[probability], str(label)]) + '\n'
      writer.write(output_line)
  return


if __name__ == "__main__":
    custom_pred_fn(MODEL_DIR)
