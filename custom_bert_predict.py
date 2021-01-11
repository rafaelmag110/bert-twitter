import os
import re
import math
import json

import pandas as pd
import numpy as np
import tensorflow as tf

from official.nlp.bert import configs as bert_configs
from official.nlp.bert import run_classifier
from official.nlp.bert import bert_models
from official.nlp.modeling import models
from official.utils.misc import distribution_utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

BERT_DIR = '/bert_env/bert_models/uncased_L-12_H-768_A-12'
DATA_DIR = '/bert_env/data'
RECORDS_DIR = '/bert_env/records'
MODEL_DIR = '/bert_env/output_dir'

EVAL_BATCH_SIZE = 32
MAX_SEQ_LENGTH = 64

gold_file = '/bert_env/data/test.tsv'
test_results_file = MODEL_DIR + '/test_results.tsv'

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
      output_line = '\t'.join([str(probability), str(label)]) + '\n'
      writer.write(output_line)
  return


def load_gold_file(input_file):
    '''
    Line structure is "sentiment \t tweet"
    '''
    senti_dict = {
        'negative':0,
        'neutral':1,
        'positive':2,
    }

    #file_path = os.path.expanduser(input_file) 
    file_path = input_file
    sentiments = []
    with open(file_path, 'r') as file:
        for line in file:
            tokens = re.split(r'\t',line)
            sentiments.append(senti_dict[tokens[0]])
    gold = np.array(sentiments)

    return gold

def load_probabilities_results(input_file):
    results = pd.read_csv(input_file,
                           sep="\t",
                           header=None,
                           names=["negative", "neutral", "positive"])
    probabilities = results.to_numpy()
    predictions = probabilities.argmax(axis=1)
    return predictions, probabilities

def load_predictions_results(input_file):
    #file_path = os.path.expanduser(input_file)
    file_path = input_file
    results, labels = [], []
    with open(file_path, "r") as re_file:
        for line in re_file:
            tokens = re.split(r'\t', line)
            results.append(int(tokens[0]))
            labels.append(int(tokens[1]))
    predictions = np.array(results)
    gold_labels = np.array(labels)

    return predictions, gold_labels

def compute_semeval_metrics(gold, predictions):

    def _multilabel_recall(index, cmtx):
        '''
        Recall is defined as the proportion between correctly classified relevant classes and
        all the known relevant classes.
        recall = TP / TP + FN
        '''
        true_gold = cmtx.iloc[index, index]
        all_gold = np.sum(cmtx.iloc[index,:].to_numpy())
        return true_gold / all_gold

    def _multilabel_precision(index, cmtx):
        '''
        Precision is defined as the proportion between correctly classified cases and all the classified cases of
        class.
        recall = TP / TP + FP
        '''
        true_pred = cmtx.iloc[index, index]
        false_pred = np.sum(cmtx.iloc[:,index].to_numpy())
        return true_pred / false_pred

    cmtx = pd.DataFrame(
        confusion_matrix(gold, predictions, labels=[0,1,2]),
        index=['gold:negative', 'gold:neutral', 'gold:positive'],
        columns=['pred:negative', 'pred:neutral', 'pred:positive']
    )

    #accuracy
    acc = accuracy_score(gold, predictions)

    #recall
    negative_recall = _multilabel_recall(0, cmtx)
    neutral_recall = _multilabel_recall(1, cmtx)
    positive_recall = _multilabel_recall(2, cmtx)
    avg_r = (negative_recall + neutral_recall + positive_recall) / 3

    #precision
    negative_precision = _multilabel_precision(0, cmtx)
    positive_precision = _multilabel_precision(2, cmtx)

    #f1
    negative_f1 = (2*negative_precision*negative_recall) / (negative_precision+negative_recall)
    positive_f1 = (2*positive_precision*positive_recall) / (positive_precision+positive_recall)
    f1_pn = (positive_f1 + negative_f1) / 2


    print('*******CONFUSION MATRIX*******')
    print(cmtx)
    print('*******EVALUATION METRICS********')
    print('Average recall: ', avg_r)
    print('F1_pn = ', f1_pn)
    print("Accuracy: ", acc)

def evaluate_test_file(test_results_file, mode='probs'):
    gold = load_gold_file(gold_file)
    print("Loaded {} test values.".format(gold.shape[0]))
    if mode == 'probs':
        predictions, _ = load_probabilities_results(test_results_file)
        print("Loaded {} predictions.".format(predictions.shape[0]))
    elif mode == 'preds':
        predictions, predicted_labels = load_predictions_results(test_results_file)
        print("Loaded {} predictions.".format(predictions.shape[0]))
        check = accuracy_score(gold, predicted_labels)
        if check == 1:
            print("The labels match.")
    compute_semeval_metrics(gold, predictions)

if __name__ == "__main__":
    custom_pred_fn(MODEL_DIR)
    evaluate_test_file(test_results_file, mode='preds')
