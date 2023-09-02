EVAL_FILE=dataset/2023_train_topics.json

# T5-NTR with no ptkbs
# for N_HISTORY in 1 3 5; do
#     for N_RESPONSES in 0 1; do
#         python3 generate_ikat.py \
#             --model_name castorini/t5-base-canard \
#             --model_path castorini/t5-base-canard \
#             --input_file ${EVAL_FILE} \
#             --output_jsonl results/ikat_train/t5ntr_history_${N_HISTORY}-${N_RESPONSES}.prediction.jsonl \
#             --device cuda:0 \
#             --batch_size 4 \
#             --n_conversations ${N_HISTORY} \
#             --n_responses ${N_RESPONSES} \
#             --num_beams 5 \
#             --max_src_length 512 \
#             --max_tgt_length 512
#     done
# done

# T5-NTR with selected ptkbs
for N_HISTORY in 1 3 5; do
    for N_RESPONSES in 0 1; do
        python3 generate_ikat.py \
            --model_name castorini/t5-base-canard \
            --model_path castorini/t5-base-canard \
            --input_file ${EVAL_FILE} \
            --output_jsonl results/ikat_train/t5ntr_history_${N_HISTORY}-${N_RESPONSES}.ptkb_select.jsonl \
            --device cuda:0 \
            --batch_size 4 \
            --n_conversations ${N_HISTORY} \
            --n_responses ${N_RESPONSES} \
            --num_beams 5 \
            --select_ptkb_as_conversation \
            --max_src_length 512 \
            --max_tgt_length 512
    done
done

# for N_HISTORY in 3 5 8 10; do
#     # selected ptkbs
#     python3 generate_ikat.py \
#         --model_name google/flan-t5-base \
#         --model_path models/ckpt/function-base-flatten/checkpoint-20000 \
#         --instruction_prefix 'Rewrite the user query based on the previous user-system conversation. user query: {} conversation: ' \
#         --conversation_prefix 'user: {0} system: {1}' \
#         --input_file ${EVAL_FILE} \
#         --output_jsonl results/ikat_train/function_flat.ptkb_none.history_${N_HISTORY}.jsonl \
#         --device cuda:2 \
#         --batch_size 2 \
#         --n_conversations ${N_HISTORY} \
#         --num_beams 5 \
#         --max_src_length 256 \
#         --max_tgt_length 64
# done
