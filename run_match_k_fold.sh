#!/bin/bash

for i in 1 2 3 4 5
do

CURRENT_DIR=`pwd`
echo "CURRENT_DIR: $CURRENT_DIR"

echo "==============================================="
echo "test k-fold ${i}..."
echo "==============================================="

# export BERT_MODEL="/home/zhuminghao/work/model/pt/RoBERTa-wwm-ext-large-Chinese/bert_chinese_large"
export BERT_MODEL="/home/zhuminghao/work/model/pt/bert-base-uncased/"
export DATA_DIR="data/clean/k_fold/k_fold${i}/"
#export DATA_DIR="data/clean/tmp/k_fold/k_fold${i}/"
export OUTPUT_DIR="output/k_fold/k_fold${i}/"
export NUM_EPOCHS=6
export MAX_SEQ_LENGTH=256
export TRAIN_BATCH_SIZE=32
export EVAL_BATCH_SIZE=32
export LEARNING_RATE=2e-5
export EVAL_STEP=900
export SEED=42

# check output directory if existed else new
if [ ! -d $OUTPUT_DIR ];then
  mkdir -p $OUTPUT_DIR
else
  rm -rf $OUTPUT_DIR/*
fi

# run traning
python3 run_match.py --model $BERT_MODEL \
--data_dir $DATA_DIR \
--output_dir $OUTPUT_DIR \
--num_train_epochs $NUM_EPOCHS \
--max_seq_length $MAX_SEQ_LENGTH \
--train_batch_size $TRAIN_BATCH_SIZE \
--eval_batch_size $EVAL_BATCH_SIZE \
--learning_rate $LEARNING_RATE \
--eval_step $EVAL_STEP \
--seed $SEED \

done