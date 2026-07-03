#!/bin/bash

# https://github.com/IINemo/lm-polygraph/blob/main/examples/configs/polygraph_eval_xsum.yaml

MODEL=Phi-3-mini-4k-instruct # Llama-2-7b-chat, Llama-2-13b-chat, Llama-2-70b-chat-8bit, Meta-Llama-3.1-8B-Instruct, Phi-3-mini-4k-instruct, gemma-2-2b-it, Mistral-Nemo-Instruct-2407
DATASET=(
    xsum
    # xsum
    # aeslc
    # aeslc
) # xsum, aeslc
METRIC=(
    rougel
    bertscore
    rougel
    bertscore
) # rougel, bertscore
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
SUFFIX=''
NUM_GENERATIONS=10
TEMPERATURE=1.0
SEED=42
LOG_DIR=logs
mkdir -p $LOG_DIR

# Get length
length=${#DATASET[@]}

for (( i=0; i<$length; i++ ))
do
    # By default, also compute uncertainty and analyze results
    # Get the current time in the format YYYY-MM-DD_HH-MM-SS
    (current_time=$(date +"%Y-%m-%d_%H-%M-%S")
    if [[ ${DATASET[$i]} == "xsum" ]]; then
        MAX_NEW_TOKEN=56
    else
        MAX_NEW_TOKEN=31
    fi

    CMD="CUDA_VISIBLE_DEVICES=${DEVICE[$i]} python snne/generate_answers.py \
        --model_name=$MODEL \
        --dataset=${DATASET[$i]} \
        --metric=${METRIC[$i]} \
        --num_generations=$NUM_GENERATIONS \
        --no-compute_snn \
        --no-compute_wsnn \
        --temperature=$TEMPERATURE \
        --reset_seed \
        --brief_prompt ${DATASET[$i]} \
        --prompt_type ${DATASET[$i]} \
        --random_seed $SEED \
        --model_max_new_tokens $MAX_NEW_TOKEN \
        --token_limit 8192 \
        --no-condition_on_question \
        --num_few_shot 0 \
        --p_true_num_fewshot 0 \
        --suffix=$SUFFIX"

    eval $CMD 2>&1 | tee ${LOG_DIR}/${current_time}_${DATASET[$i]}_${METRIC[$i]}_device${DEVICE[$i]}.log) &
done