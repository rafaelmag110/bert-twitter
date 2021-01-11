# SCRIPT NOT WORKING PROPERLY

export DATA_DIR=/bert_env/data
export RECORDS_DIR=/bert_env/records
export BERT_DIR=/bert_env/bert_models/uncased_L-12_H-768_A-12
export MODEL_DIR=/bert_env/output_dir

python -m official.nlp.bert.run_classifier \
--mode='predict' \
--input_meta_data_path=${DATA_DIR}/input_meta_data \
--eval_data_path=${RECORDS_DIR}/test.tf_record \
--bert_config_file=${BERT_DIR}/bert_config.json \
#--predict_checkpoint_path=${MODEL_DIR}/ctl_step_100000.ckpt-12 \
--model_dir=${MODEL_DIR}/ \
--eval_batch_size=32 \
--distribution_strategy=mirrored
