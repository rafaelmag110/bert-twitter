import os
import json

import numpy as np
import tensorflow as tf

from official.nlp.bert import tokenization
from official.nlp.data import classifier_data_lib

print('=====Creating InputFeatures records.')
print('Tensorflow version:', tf.__version__)

#PATHS definition

DATA_DIR="/bert_env/data"
RECORDS_DIR="/bert_env/records"
BERT_DIR="/bert_env/bert_models/uncased_L-12_H-768_A-12"
# TPU_ADDRESS='grpc://'+tpu.cluster_spec().as_dict()['worker'][0]


#VARIABLE DEFINITION
MAX_SEQ_LENGTH=128

train_data_output_path="/bert_env/records/train.tf_record"
eval_data_output_path="/bert_env/records/eval.tf_record"
test_data_output_path="/bert_env/records/test.tf_record"


# InputFeatures have to be generated for model input

# Definition for the twitter data processor 

class TwitterProcessor(classifier_data_lib.DataProcessor):
  """Processor for the Twitter data """

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train_clean.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test_clean.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["negative","neutral", "positive"]

  @staticmethod
  def get_processor_name():
    """See base class."""
    return "TWITTER"

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      # print(line)
      # if i == 0:
      #   continue
      guid = "%s-%s" % (set_type, i)
      if len(line) == 2:
        text_a = self.process_text_fn(line[1])
      else:
        continue
      if set_type == "test":
        label = "neutral"
      else:
        label = self.process_text_fn(line[0])
      examples.append(
          classifier_data_lib.InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

processor=TwitterProcessor(process_text_fn=tokenization.convert_to_unicode)
tokenizer = tokenization.FullTokenizer(
                vocab_file=BERT_DIR + "/vocab.txt", 
                do_lower_case=True)


print('==Creating input meta data.')
# Generate tf records for input to the model
input_meta_data = classifier_data_lib.generate_tf_record_from_data_file(processor= processor,
                                                                        data_dir=DATA_DIR,
                                                                        tokenizer=tokenizer,
                                                                        train_data_output_path=train_data_output_path,
                                                                        eval_data_output_path=eval_data_output_path,
                                                                        max_seq_length=MAX_SEQ_LENGTH)
print(input_meta_data)
with open("/bert_env/data/input_meta_data", "w") as writer:
  json.dump(input_meta_data, writer)

print('======Done.')
