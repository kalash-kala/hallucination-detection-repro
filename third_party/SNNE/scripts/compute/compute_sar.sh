#!/bin/bash

LIST_DATSET=(
    squad
    # trivia_qa
    # nq
    # svamp
    # bioasq
)
LIST_METRIC=(
    squad
    # squad
    # squad
    # squad
    # squad
)
LIST_GPU=(
    0
    1
    2
    3
    4
    5
    6
    7
)
# Change the run name to match your wandb run
LIST_RUN_NAME=(
    run-20250606_154223-ucj1spxh
)
BASE_PATH=./$USER/uncertainty/wandb
NUM_GENERATIONS=10
MODEL_NAME=Meta-Llama-3.1-8B-Instruct
THRESHOLD=0.5
CONCAT=False
ENTAILMENT=cross-encoder/stsb-roberta-large

for i in "${!LIST_DATSET[@]}"; do
    (DATASET="${LIST_DATSET[i]}"
    RUN_NAME="${LIST_RUN_NAME[i]}"
    DATA_PATH=${BASE_PATH}/${RUN_NAME}/files
    METRIC="${LIST_METRIC[i]}"
    ENTAILMENT_NAME=$(basename $ENTAILMENT)
    SUFFIX=_${ENTAILMENT_NAME}_${METRIC}_thr${THRESHOLD}_reset_seed_temp1.0
    GPU="${LIST_GPU[i]}"

    CMD="CUDA_VISIBLE_DEVICES=$GPU python snne/compute_sar.py \
        --dataset=$DATASET \
        --num_generations=$NUM_GENERATIONS \
        --model_name=$MODEL_NAME \
        --data_path=$DATA_PATH \
        --suffix=$SUFFIX \
        --metric=$METRIC \
        --entailment_model=$ENTAILMENT \
        --metric_threshold=$THRESHOLD"

    if [[ $CONCAT == "False" ]]; then
        CMD="$CMD --no-condition_on_question"
    fi

    eval $CMD) &
done