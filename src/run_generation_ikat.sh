EVAL_FILE=dataset/2023_train_topics.json

# generate
python3 generate.py \
    --model_name castorini/t5-base-canard \
    --model_path castorini/t5-base-canard \
    --input_file ${EVAL_FILE} \
    --output_jsonl results/ikat_train.ntr.prediction.jsonl \
    --device cuda \
    --batch_size 4 \
    --n_conversations 3 \
    --num_beams 5 \
    --max_src_length 512 \
    --max_tgt_length 32
