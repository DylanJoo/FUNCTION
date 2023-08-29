EVAL_FILE=dataset/2023_train_topics.json
export CUDA_VISIBLE_DEVICES=1

# # T5-NTR with no ptkbs
# for N_HISTORY in 3 5 8; do
#     python3 generate_ikat.py \
#         --model_name castorini/t5-base-canard \
#         --model_path castorini/t5-base-canard \
#         --input_file ${EVAL_FILE} \
#         --output_jsonl results/ikat_train/t5ntr.ptkb_none.history_${N_HISTORY}.jsonl \
#         --device cuda \
#         --batch_size 4 \
#         --n_conversations ${N_HISTORY} \
#         --include_response \
#         --num_beams 5 \
#         --max_src_length 512 \
#         --max_tgt_length 128
# done

# # T5-NTR with selected ptkbs
# for N_HISTORY in 3 5 8; do
#     # selected ptkbs
#     python3 generate_ikat.py \
#         --model_name castorini/t5-base-canard \
#         --model_path castorini/t5-base-canard \
#         --input_file ${EVAL_FILE} \
#         --output_jsonl results/ikat_train/t5ntr.ptkb_select.history_${N_HISTORY}.jsonl \
#         --device cuda \
#         --batch_size 4 \
#         --include_response \
#         --n_conversations ${N_HISTORY} \
#         --num_beams 5 \
#         --max_src_length 512 \
#         --max_tgt_length 128 \
#         --select_ptkb_as_conversation 
# done

# T5-NTR with all ptkbs
for N_HISTORY in 3 5 8; do
    # selected ptkbs
    python3 generate_ikat.py \
        --model_name castorini/t5-base-canard \
        --model_path castorini/t5-base-canard \
        --input_file ${EVAL_FILE} \
        --output_jsonl results/ikat_train/t5ntr.ptkb_all.history_${N_HISTORY}.jsonl \
        --device cuda \
        --batch_size 4 \
        --n_conversations ${N_HISTORY} \
        --num_beams 5 \
        --max_src_length 512 \
        --max_tgt_length 128\
        --all_ptkb_as_conversation 
done

# for N_HISTORY in 3 5 8 10; do
#     # selected ptkbs
#     python3 generate_ikat.py \
#         --model_name google/flan-t5-base \
#         --model_path models/ckpt/function-base-flatten/checkpoint-20000 \
#         --instruction_prefix 'Rewrite the query based on the user-system conversation. query: {} conversation: ' \
#         --conversation_prefix 'user: {0} system: {1}' \
#         --input_file ${EVAL_FILE} \
#         --output_jsonl results/ikat_train/function_flat.ptkb_none.history_${N_HISTORY}.jsonl \
#         --device cuda \
#         --batch_size 4 \
#         --n_conversations ${N_HISTORY} \
#         --num_beams 5 \
#         --max_src_length 256 \
#         --max_tgt_length 64
# done
