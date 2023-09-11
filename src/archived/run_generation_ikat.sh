EVAL_FILE=dataset/2023_train_topics.json
TEST_FILE=dataset/2023_test_topics.json

# T5-NTR with no ptkbs
# for N_HISTORY in 1 3 5; do
#     for N_RESPONSES in 0 1 2 3; do
#         python3 generate_ikat.py \
#             --model_name castorini/t5-base-canard \
#             --model_path castorini/t5-base-canard \
#             --input_file ${EVAL_FILE} \
#             --output_jsonl results/ikat_train/t5ntr_history_${N_HISTORY}-${N_RESPONSES}.jsonl \
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
# for N_HISTORY in 0; do
#     for N_RESPONSES in 0; do
#         python3 generate_ikat.py \
#             --model_name castorini/t5-base-canard \
#             --model_path castorini/t5-base-canard \
#             --input_file ${EVAL_FILE} \
#             --output_jsonl results/ikat_train/t5ntr_history_${N_HISTORY}-${N_RESPONSES}.ptkb_select.jsonl \
#             --device cuda:0 \
#             --batch_size 4 \
#             --n_conversations ${N_HISTORY} \
#             --n_responses ${N_RESPONSES} \
#             --num_beams 5 \
#             --select_ptkb_as_conversation \
#             --max_src_length 512 \
#             --max_tgt_length 512
#     done
# done


# Function-base-flatten
# for N_HISTORY in 10; do
#     # selected ptkbs
#     python3 generate_ikat.py \
#         --model_name google/flan-t5-base \
#         --model_path models/ckpt/function-base-flatten/checkpoint-20000 \
#         --instruction_prefix 'Rewrite the user query based on previous user-system conversations. user query: {} conversation: ' \
#         --conversation_prefix 'user: {0} system: {1}' \
#         --input_file ${TEST_FILE} \
#         --output_jsonl results/ikat_test/function_flat.history_${N_HISTORY}.jsonl \
#         --device cuda:0 \
#         --batch_size 1 \
#         --n_conversations ${N_HISTORY} \
#         --num_beams 5 \
#         --max_src_length 512 \
#         --max_tgt_length 128 \
#         --all_ptkb_as_conversation
# done

# Function-base-flatten-sep
for N_HISTORY in 10; do
    # selected ptkbs
    python3 generate_ikat.py \
        --model_name google/flan-t5-base \
        --model_path models/ckpt/function-base-flatten-sep/checkpoint-20000 \
        --conversation_prefix 'rewrite the user utterance: {0} based on the conversation: user: {1} system: {2}' \
        --input_file ${TEST_FILE} \
        --output_jsonl results/ikat_test/function_flat_sep.history_${N_HISTORY}.jsonl \
        --device cuda:0 \
        --batch_size 1 \
        --n_conversations ${N_HISTORY} \
        --num_beams 5 \
        --max_src_length 512 \
        --max_tgt_length 128 \
        --all_ptkb_as_conversation
done
# # Function-base-compressed
# for N_HISTORY in 3 5 8 10; do
#     # selected ptkbs
#     python3 generate_ikat.py \
#         --model_name google/flan-t5-base \
#         --model_path models/ckpt/function-base-compressed/checkpoint-20000 \
#         --instruction_prefix 'Rewrite the user query based on previous user-system conversations. user query: {} conversation: ' \
#         --conversation_prefix 'user: {0} system: {1}' \
#         --input_file ${EVAL_FILE} \
#         --output_jsonl results/ikat_train/function_comp.ptkb_all.history_${N_HISTORY}.jsonl \
#         --device cuda:2 \
#         --batch_size 2 \
#         --n_conversations ${N_HISTORY} \
#         --num_beams 5 \
#         --max_src_length 128 \
#         --max_tgt_length 64 \
#         --max_src_conv_length 256
#         --all_ptkb_as_conversation
# done
