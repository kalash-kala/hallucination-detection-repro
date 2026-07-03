#!/bin/bash

MODEL=Meta-Llama-3.1-8B-Instruct # Llama-2-7b-chat, Llama-2-13b-chat, Meta-Llama-3.1-8B-Instruct, Phi-3-mini-4k-instruct, gemma-2-2b-it, Mistral-Nemo-Instruct-2407
DATASET=(
    squad
    # trivia_qa
    # nq
    # svamp
    # bioasq
)
DEVICE=(
    0
    1
    2
    3
    4
    5
    6
    7
)
MINP=0.0
CONCAT=True
NUM_GENERATIONS=10
TEMPERATURE=1.0
SUFFIX=""
SEED=10
LOG_DIR=logs
mkdir -p $LOG_DIR

# Get length
length=${#DATASET[@]}

for (( i=0; i<$length; i++ ))
do
    # By default, also compute uncertainty and analyze results
    # Get the current time in the format YYYY-MM-DD_HH-MM-SS
    current_time=$(date +"%Y-%m-%d_%H-%M-%S")
    (CMD="CUDA_VISIBLE_DEVICES=${DEVICE[$i]} python snne/generate_answers.py \
        --model_name=$MODEL \
        --dataset=${DATASET[$i]} \
        --metric=squad \
        --num_generations=$NUM_GENERATIONS \
        --no-compute_snn \
        --no-compute_wsnn \
        --temperature=$TEMPERATURE \
        --min_p=$MINP \
        --reset_seed \
        --random_seed $SEED \
        --suffix=$SUFFIX"
    
    if [[ $CONCAT == "False" ]]; then
        CMD="$CMD --no-condition_on_question"
    fi

    eval $CMD 2>&1 | tee ${LOG_DIR}/${current_time}_${DATASET[$i]}_squad_device${DEVICE[$i]}.log) &
done