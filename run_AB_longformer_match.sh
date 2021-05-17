#!/bin/bash

for val in "A" "B"
do

CURRENT_DIR=`pwd`
echo "CURRENT_DIR: $CURRENT_DIR"

echo "==============================================="
echo "Train  ${val}"
echo "==============================================="

export BERT_MODEL="/home/zhuminghao/work/model/pt/longformer/"
#export BERT_MODEL="/home/zhuminghao/work/model/pt/bert-base-uncased/"
#export DATA_DIR="data/clean/round2/${val}/"
export DATA_DIR="data/clean/round2_rept/${val}/"
#export DATA_DIR="data/clean/tmp/round2/${val}/"
#export OUTPUT_DIR="output/round2/${val}/"
export OUTPUT_DIR="output/round2_rept/${val}/"
#export OUTPUT_DIR="output/round2/${val}/"
export NUM_EPOCHS=4
export MAX_SEQ_LENGTH=1024
export TRAIN_BATCH_SIZE=4
export EVAL_BATCH_SIZE=4
export LEARNING_RATE=5e-5
export EVAL_STEP=2000
export SEED=42

# check output directory if existed else new
if [ ! -d $OUTPUT_DIR ];then
  mkdir -p $OUTPUT_DIR
#else
#  rm -rf $OUTPUT_DIR/*
fi

# run traning
python3 run_longformer_match.py --model $BERT_MODEL \
--data_dir $DATA_DIR \
--output_dir $OUTPUT_DIR \
--num_train_epochs $NUM_EPOCHS \
--max_seq_length $MAX_SEQ_LENGTH \
--train_batch_size $TRAIN_BATCH_SIZE \
--eval_batch_size $EVAL_BATCH_SIZE \
--learning_rate $LEARNING_RATE \
--eval_step $EVAL_STEP \
--seed $SEED
#--do_adversarial


#echo "==============================================="
#echo "predict ${val}"
#echo "==============================================="
#
##export DATA_DIR="data/clean/AB/${val}/"
##export OUTPUT_DIR="output/AB/${val}/"
##export MAX_SEQ_LENGTH=256
#export TEST_BATCH_SIZE=512
##export SEED=42
#
## run predict
#python3 run_match_predict.py --data_dir $DATA_DIR \
#--output_dir $OUTPUT_DIR \
#--max_seq_length $MAX_SEQ_LENGTH \
#--test_batch_size $TEST_BATCH_SIZE \

done