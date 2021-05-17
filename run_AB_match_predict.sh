#!/bin/bash

for val in "A" 'B'
do

CURRENT_DIR=`pwd`
echo "CURRENT_DIR: ${CURRENT_DIR}"

echo "==============================================="
echo "predict ${val}"
echo "==============================================="

export DATA_DIR="data/clean/round2_rept/${val}/"
export OUTPUT_DIR="output/round2_rept/${val}/"
export MAX_SEQ_LENGTH=256
export TEST_BATCH_SIZE=256
export SEED=42

# run predict
python3 run_match_predict.py \
--data_dir $DATA_DIR \
--output_dir $OUTPUT_DIR \
--max_seq_length $MAX_SEQ_LENGTH \
--test_batch_size $TEST_BATCH_SIZE \

done