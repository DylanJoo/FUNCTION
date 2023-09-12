EVAL_FILE=/home/jhju/datasets/qrecc/qrecc_test.json
MY_BASELINE=DylanJHJ/ntr-base-qrecc

mkdir -p results/qrecc_test
# T5 NTR baseline
# n-m contains n user's utterances and m system's
for N_HISTORY in 3 5 7 10; do
    for N_RESPONSES in 3; do
        python3 generate.py \
                --model_name castorini/t5-base-canard \
                --model_path ${MY_BASELINE} \
                --input_file ${EVAL_FILE} \
                --output_jsonl results/qrecc_test/t5ntr_history_${N_HISTORY}-${N_RESPONSES}.prediction.jsonl \
                --device cuda \
                --batch_size 16 \
                --n_conversations ${N_HISTORY} \
                --n_responses ${N_RESPONSES} \
                --num_beams 5 \
                --max_src_length 512 \
                --max_tgt_length 32
    done
done

# Funtion-base-flatten. 
# n-m contains n user's utterances and m system's
for N_HISTORY in 3 5 7 10; do
    python3 generate.py \
        --model_name google/flan-t5-base \
        --model_path models/ckpt/function-base-flatten/checkpoint-20000 \
        --input_file ${EVAL_FILE} \
        --output_jsonl results/qrecc_test/function_flat.history_${N_HISTORY}.prediction.jsonl \
        --device cuda \
        --batch_size 16 \
        --instruction_prefix 'Rewrite the user utterance based on previous user-system conversations. user: {}' \
        --conversation_prefix 'conversation: user: {0} system: {1}' \
        --n_conversations ${N_HISTORY} \
        --num_beams 5 \
        --max_src_length 512 \
        --max_tgt_length 32
done
