EVAL_FILE=dataset/2023_train_topics.json

export CUDA_VISIBLE_DEVICES=1
# generate
python3 generate_ikat.py \
    --model_name castorini/t5-base-canard \
    --model_path castorini/t5-base-canard \
    --input_file ${EVAL_FILE} \
    --output_jsonl results/ikat_train.ntr.prediction.jsonl \
    --device cuda \
    --batch_size 2 \
    --n_conversations 3 \
    --num_beams 5 \
    --max_src_length 512 \
    --max_tgt_length 32 \
    --select_ptkb_as_conversation
