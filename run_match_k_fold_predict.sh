#!/bin/bash

for i in 1 2 3 4 5
do

CURRENT_DIR=`pwd`
echo "CURRENT_DIR: $CURRENT_DIR"

echo "==============================================="
echo "test k-fold ${i}..."
echo "==============================================="

export DATA_DIR="data/clean/k_fold/k_fold${i}/"
export OUTPUT_DIR="output/k_fold/k_fold${i}/"
export MAX_SEQ_LENGTH=256
export TEST_BATCH_SIZE=2048
export SEED=42

# run traning
python3 run_match_k_fold_predict.py \
--data_dir $DATA_DIR \
--output_dir $OUTPUT_DIR \
--max_seq_length $MAX_SEQ_LENGTH \
--test_batch_size $TEST_BATCH_SIZE \
--seed $SEED \

done