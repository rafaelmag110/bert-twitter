export BERT_BASE_DIR=/bert_env/bert_models/uncased_L-12_H-768_A-12
export OUTPUT_DIR=/bert_env/output_dir
export RECORD_DIR=/bert_env/records
export DATA_DIR=/bert_env/data
# export MODEL_FIT_DIR=/bert_env/model_fit

echo '======PRETRAINING BERT=======' > output_pretraining.txt

python -m official.nlp.bert.run_pretraining \
--input_files=${RECORD_DIR}/tf_examples.tfrecord \
--model_dir=${OUTPUT_DIR}/ \
--bert_config_file=${BERT_BASE_DIR}/bert_config.json \
--init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
--train_batch_size=32 \
--max_seq_length=128 \
--max_predictions_per_seq=3 \
--num_train_epochs=10 \
--num_steps_per_epoch=10000 \
--warmup_steps=10000 \
--learning_rate=2e-5 >> output_pretraining.txt
