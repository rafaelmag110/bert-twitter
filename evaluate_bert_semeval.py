import os
import re

import pandas as pd
import numpy as np


from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

MODEL_DIR = 'output_dir'

gold_file = 'data/test.tsv'
test_results_file = MODEL_DIR + '/test_results.tsv'

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
    evaluate_test_file(test_results_file, mode='preds')