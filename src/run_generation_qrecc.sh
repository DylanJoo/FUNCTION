EVAL_FILE=/home/jhju/datasets/qrecc/qrecc_test.json

mkdir -p results/qrecc_test
MODEL_NAME=google/flan-t5-base
MODEL_NAME=castorini/t5-base-canard

MODEL_PATH=models/ckpt/ntr-base/checkpoint-15000
MODEL_PATH=castorini/t5-base-canard

# T5 NTR baseline
# n-m contains n user's utterances and m system's
for N_HISTORY in 3 6; do
    for N_RESPONSES in 3; do
        python3 generate.py \
                --model_name ${MODEL_NAME} \
                --model_path ${MODEL_PATH} \
                --input_file ${EVAL_FILE} \
                --output_jsonl results/qrecc_test/t5ntr_history_${N_HISTORY}-${N_RESPONSES}.jsonl \
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
# for N_HISTORY in 3 6; do
#     python3 generate.py \
#         --model_name google/flan-t5-base \
#         --model_path models/ckpt/function-base-flatten/checkpoint-15000 \
#         --input_file ${EVAL_FILE} \
#         --output_jsonl results/qrecc_test/function_flat.history_${N_HISTORY}.jsonl \
#         --device cuda \
#         --batch_size 16 \
#         --instruction_prefix 'Based on previous conversations, rewrite the user utterance: {} into a standalone query.' \
#         --conversation_prefix 'user: {0} system: {1}' \
#         --n_conversations ${N_HISTORY} \
#         --num_beams 5 \
#         --max_src_length 512 \
#         --max_tgt_length 32
# done
