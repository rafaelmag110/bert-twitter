export BERT_BASE_DIR=/bert_env/bert_models/uncased_L-12_H-768_A-12
export RECORD_DIR=/bert_env/records
export DATA_DIR=/bert_env/data
# export MODEL_FIT_DIR=/bert_env/model_fit

echo '======Creating Pretraining data=======' > output_creating_prdata.txt

python -m official.nlp.data.create_pretraining_data \
--input_file=${DATA_DIR}/bert_pt.txt \
--output_file=${RECORD_DIR}/tf_examples.tfrecord \
--vocab_file=${BERT_BASE_DIR}/vocab.txt \
--do_lower_case=True \
--max_seq_length=128 \
--max_predictions_per_seq=3 \
--masked_lm_prob=0.15 \
--random_seed=12345 \
--dupe_factor=5 >> output_creating_prdata.txt
