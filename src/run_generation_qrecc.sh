EVAL_FILE=/home/jhju/datasets/qrecc/qrecc_test.json

# n-m contains n user's utterances and m system's
N_HISTORY=5
for N_RESPONSES in 1 2 3; do
    python3 generate.py \
            --model_name castorini/t5-base-canard \
            --model_path castorini/t5-base-canard \
            --input_file ${EVAL_FILE} \
            --output_jsonl results/qrecc_test/t5ntr_history_${N_HISTORY}-${N_RESPONSES}.prediction.jsonl \
            --device cuda:0 \
            --batch_size 8 \
            --n_conversations ${N_HISTORY} \
            --n_responses ${N_RESPONSES} \
            --num_beams 5 \
            --max_src_length 512 \
            --max_tgt_length 32
done

# n-m contains n user's utterances and m system's
N_HISTORY=5
for N_RESPONSES in 1 2 3; do
    python3 generate.py \
            --model_name castorini/t5-base-canard \
            --model_path castorini/t5-base-canard \
            --input_file ${EVAL_FILE} \
            --output_jsonl results/qrecc_test/t5ntr_history_${N_HISTORY}-${N_RESPONSES}.prediction.jsonl \
            --device cuda:0 \
            --batch_size 8 \
            --n_conversations ${N_HISTORY} \
            --n_responses ${N_RESPONSES} \
            --num_beams 5 \
            --max_src_length 512 \
            --max_tgt_length 32
done

