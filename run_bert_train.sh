echo '=====OUTPUT FOR BERT_TRAIN SCRIPT=====' > output.txt

export DATA_DIR=/bert_env/data
export RECORDS_DIR=/bert_env/records
export BERT_DIR=/bert_env/bert_models/uncased_L-12_H-768_A-12
export MODEL_DIR=/bert_env/output_dir

python -m official.nlp.bert.run_classifier \
--mode='train_and_eval' \
--input_meta_data_path=${DATA_DIR}/input_meta_data \
--train_data_path=${RECORDS_DIR}/train.tf_record \
--eval_data_path=${RECORDS_DIR}/eval.tf_record \
--bert_config_file=${BERT_DIR}/bert_config.json \
--init_checkpoint=${BERT_DIR}/bert_model.ckpt  \
--model_dir=${MODEL_DIR}/ \
--train_batch_size=32 \
--eval_batch_size=32 \
--steps_per_loop=1 \
--learning_rate=2e-5 \
--num_train_epochs=5 \
--distribution_strategy=mirrored >> output.txt
