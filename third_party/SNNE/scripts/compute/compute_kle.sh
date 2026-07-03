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
MODEL_NAME=Meta-Llama-3.1-8B-Instruct
NUM_GENERATIONS=10
ENTAILMENT=deberta
CONCAT=True # True for QA and False for summarization and translation

for i in "${!LIST_DATSET[@]}"; do
    (DATASET="${LIST_DATSET[i]}"
    RUN_NAME="${LIST_RUN_NAME[i]}"
    DATA_PATH=${BASE_PATH}/${RUN_NAME}/files
    METRIC="${LIST_METRIC[i]}"
    SUFFIX=_${ENTAILMENT}_concat${CONCAT}_${METRIC}
    GPU="${LIST_GPU[i]}"
    CMD="CUDA_VISIBLE_DEVICES=$GPU python snne/compute_kle.py \
        --dataset=$DATASET \
        --num_generations=$NUM_GENERATIONS \
        --model_name=$MODEL_NAME \
        --data_path=$DATA_PATH \
        --entailment_model=$ENTAILMENT \
        --suffix=$SUFFIX"

    if [[ $CONCAT == "False" ]]; then
        CMD="$CMD --no-condition_on_question"
    fi

    eval $CMD) &
done