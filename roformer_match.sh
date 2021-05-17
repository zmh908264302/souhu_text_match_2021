#!/bin/bash

for i in "ssa" "ssb" "sla"
do

CURRENT_DIR=`pwd`
echo "CURRENT_DIR: $CURRENT_DIR"

echo "==============================================="
echo "test k-fold ${i}..."
echo "==============================================="

export BERT_MODEL="/mnt/model/pt/chinese_roformer_base"
export DATA_DIR="data/clean/round3/${i}/"
export OUTPUT_DIR="output/round3/${i}/"
export NUM_EPOCHS=3
export TRAIN_BATCH_SIZE=8
export EVAL_BATCH_SIZE=8
export LEARNING_RATE=2e-5
export SEED=42
if [ ${i} != "ssa" ] || [ ${i} != "ssb" ]
then
    EVAL_STEP=400
else
    EVAL_STEP=1200
fi
export EVAL_STEP


export MAX_SEQ_LENGTH=256

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