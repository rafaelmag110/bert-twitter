import os
import sys
import re

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_enum('mode', None,
                    ['check', 'remove_id_col', 'reformat', 'clean', 
                    'extract_text','sentence_segmentation'],
                    'Define the mode that the program will run.')
flags.DEFINE_boolean('all_files', False,
                    'Execute program for all files or single file.')
flags.DEFINE_string('input_file', None, 
                    'Defines the input file.')
flags.DEFINE_string('output_file', None,
                    'Defines the output file.')
flags.DEFINE_integer('text_col', 0,
                  'Defines in which column the record holds the tweet text. (0-indexed)'
                  'The default is set to 0.')

def remove_tweetId_column(input_file, output_file, sep=','):
  '''
  Some tweet datasets contain the tweetId as the first column.
  This function removes that first id column as it is mostly useless besides tweet retrieval.
  The output file will open in append mode to output multiple input files to one output file. Output file should be deleted in order to write from the start.
  '''
  count=0
  wrote_ln=0
  with open(output_file, "a") as out_file:
    with open(input_file, "r") as in_file:
      for index, line in enumerate(in_file):
        tokens = re.split(sep, line)
        if re.match(r"\d{17,18}", tokens[0]) != None:
          out_file.write(sep.join(tokens[1:]))
          wrote_ln+=1
        else:
          logging.info("{}: Token {} didn't match at line {}".format(input_file, tokens[0], index))
        count += 1
  logging.info("{}: Read {} lines, wrote {} lines".format(input_file, count, wrote_ln))
        

def check_entry_format(input_file, sep=','):
  '''
  This checks if an input file is strutured in a way that the TwitterProcessor can process.
  The twitter processor handles files in the following format:
  [class]\t[tweet]
  '''
  logging.info('Checking file: {}.'.format(input_file))

  count=0
  errors=0
  with open(input_file, "r") as file:
    for index, line in enumerate(file):
      tokens = re.split(sep, line)
      if not len(tokens) == 2:
        logging.info("{}: Tab separator not well defined at line {}".format(input_file,index))
        errors+=1
      
      sentiment = re.fullmatch(r"(positive|negative|neutral)",tokens[0])
      if not sentiment:
        logging.info("{}: Problem in line {} in class {} ".format(input_file, index, tokens))
        errors+=1
      
      #match every character except tab and newline. String must end with \n
      tweet = re.fullmatch(r"[^\t\n]+\n$", tokens[1])
      if not tweet:
        logging.info("{}: Problem in line {} in tweet:{}".format(input_file, index,tokens))
        errors+=1
      count+=1
  logging.info("{}: checked {} lines and found {} errors.".format(input_file, count, errors))

def sentence_reformating(input_file, output_file, sep=','):
  '''
  '''
  logging.info("Reformating the sentence.")

  count=0
  wrote_ln=0
  with open(output_file, "a") as output_f:
    with open(input_file, "r") as input_f:
      for index, line in enumerate(input_f):
        sentence_build = []
        tokens = re.split(sep, line)

        if re.fullmatch(r"(positive|negative|neutral)", tokens[0]) != None:
          sentence_build.append(tokens[0])

          if len(tokens) == 2:
            # tokens are well formated
            sentence_build.append(tokens[1])

          elif len(tokens) > 2 and tokens[-1] == '\n':
            #tab sep before newline
            new_tweet = tokens[1]+"\n"
            sentence_build.append(new_tweet)
          elif len(tokens) > 2 and tokens[-1] == '':
            #random tab sep at the end of the sequence
            new_tweet = tokens[1]+"\n"
            sentence_build.append(new_tweet)
          elif len(tokens) > 2:
            new_tweet = ''
            for token in tokens[1:]:
                new_tweet= new_tweet+token
            sentence_build.append(new_tweet)
            
            
          output_f.write(sep.join(sentence_build))
          wrote_ln+=1
        else:
          logging.info("Not know format in line {} with tokens {}".format(index, tokens))
        count+=1
  logging.info("{}: Read {} lines, wrote {} lines".format(input_file, count, wrote_ln))

def text_cleaner(input_file, output_file, sep=',', text_col=0):
  '''
  '''
  def _clean_text(text):
    #Regex to match
    url_regex = r'https?://[^\s]+'
    amp_regex = r'&amp;'
    username_regex = r'@[A-Za-z0-9._-]+'
    # hashtag_regex = r'#[A-Za-z0-9]+'
    hashtag_regex = r'#'

    no_url = re.sub(url_regex, '', text)
    no_url_usr = re.sub(username_regex, 'mention', no_url)
    no_url_usr_ht = re.sub(hashtag_regex, '', no_url_usr)
    # no_url_usr_ht = re.sub(r'#', '', no_url_usr)
    no_url_usr_ht_amp = re.sub(amp_regex, 'and', no_url_usr)
    return no_url_usr_ht_amp
  
  count=0
  wrote_ln=0
  with open(output_file, 'a') as file_output:
    with open(input_file, 'r') as file_input:
      for line in file_input:
        count += 1
        sentence_builder=[]
        tokens = re.split(sep, line)

        sentence_builder += tokens[:text_col]

        try:

          sentence_builder.append(_clean_text(tokens[text_col]).strip(' '))

        except IndexError:
          print("Nothing in column {}. Please check index.".format(text_col))
          break

        sentence_builder += tokens[text_col+1:]

        file_output.write(sep.join(sentence_builder))
        wrote_ln += 1
  
  logging.info("{}: Read {} lines, wrote {} lines".format(input_file, count, wrote_ln))

def text_extracter(input_file, output_file, sep=',', text_col=0):
  '''
  '''
  count=0
  wrote_ln=0
  with open(output_file, 'a') as file_output:
    with open(input_file, 'r') as file_input:
      for line in file_input:
        count += 1
        tokens = re.split(sep, line)

        try:
          file_output.write(tokens[text_col].strip(' '))
          wrote_ln += 1
        except IndexError:
          print("Nothing in column {}. Please check index.".format(text_col))
          break

  
  logging.info("{}: Read {} lines, wrote {} lines".format(input_file, count, wrote_ln))

def sentence_segmentation(input_file, output_file, sep=','):
  '''
    Input file reformating to the input format needed by the "create_pretraining_data.py" from bert. That script generates InputExemples serialized into TFRecord file formats for the MaskedLM and NSP tasks of bert pre-training.
    This formating removes the sentiment column and segments the sentences in the tweets. Each tweet is treated as a document and each document is delimited by a white line. Sentence segmentation is done using spaCy.
  '''
  import spacy
  nlp = spacy.load('en_core_web_sm')

  count = 0
  max_tokens = 0
  wrote_ln=0
  with open(output_file, 'a') as file_output:
    with open(input_file, 'r') as file_input:
      for line in file_input:
        count+=1
        if count % 10000 == 0:
          logging.info('{}: Read {} lines.'.format(input_file, count))
        tokens = re.split(sep, line)
        
        if len(tokens) == 2:
          tweet_text = tokens[1]
          tweet_text = tweet_text.strip('\n')

          doc = nlp(tweet_text)
          nm_tokens=len(doc)
          max_tokens = nm_tokens if nm_tokens > max_tokens else max_tokens
          sents = list(doc.sents)
          for sent in sents:
            file_output.write(sent.text+'\n')
            wrote_ln+=1
          file_output.write('\n')
  logging.info("{}: Read {} lines, wrote {} lines".format(input_file, count, wrote_ln))
  logging.info("{}: Max tokens in tweet: {}".format(input_file, max_tokens))

 
def main(_):
  file_list=[]

  sep_by_filetype_dict = {
    '.csv' : ',',
    '.tsv' : '\t'
  }

  def _remove_output_file(output_file):
    if os.path.exists(output_file):
        os.remove(output_file)

  if FLAGS.all_files:
    with os.scandir(".") as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith('.txt'):
                file_list.append(entry.name)
  else:
    file_list.append(FLAGS.input_file)

  if FLAGS.mode == 'check':
    #check files for correct format
    [check_entry_format(file, sep=sep_by_filetype_dict[file[-4:]]) for file in file_list]
  elif FLAGS.mode == "remove_id_col":
    #remove the tweet id column at the start of the file.
    _remove_output_file(FLAGS.output_file)
    [remove_tweetId_column(file, FLAGS.output_file, sep=sep_by_filetype_dict[file[-4:]]) for file in file_list]
  elif FLAGS.mode == 'reformat':
    #reformat the input files to correct format
    _remove_output_file(FLAGS.output_file)
    [sentence_reformating(file, FLAGS.output_file, sep=sep_by_filetype_dict[file[-4:]]) for file in file_list]
  elif FLAGS.mode == 'clean':
    #clean the text 
    _remove_output_file(FLAGS.output_file)
    [text_cleaner(file, FLAGS.output_file, sep=sep_by_filetype_dict[file[-4:]], text_col=FLAGS.text_col) for file in file_list]
  elif FLAGS.mode == 'sentence_segmentation':
    #clean the text 
    _remove_output_file(FLAGS.output_file)
    [sentence_segmentation(file, FLAGS.output_file, sep=sep_by_filetype_dict[file[-4:]]) for file in file_list]
  elif FLAGS.mode == 'extract_text':
    #extract text
    _remove_output_file(FLAGS.output_file)
    [text_extracter(file, FLAGS.output_file, sep=sep_by_filetype_dict[file[-4:]], text_col=FLAGS.text_col) for file in file_list]

if __name__ == "__main__":
  flags.mark_flag_as_required('mode')
  flags.mark_flag_as_required('input_file')
  app.run(main)
