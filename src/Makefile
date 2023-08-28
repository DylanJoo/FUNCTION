TRAIN_FILE=/tmp2/jhju/personalized.conv.search/data/qrecc/qrecc_train.json
export CUDA_VISIBLE_DEVICES=1
train_flatten:
	python3 train_flatten.py \
     		--model_name_or_path google/flan-t5-base \
     		--tokenizer_name google/flan-t5-base \
		--config_name google/flan-t5-base \
     		--train_file ${TRAIN_FILE} \
		--output_dir models/ckpt/FUNCTION-base-flatten \
	        --per_device_train_batch_size 8 \
	        --max_src_length 128 \
	        --max_tgt_length 32 \
	        --learning_rate 1e-4 \
	        --evaluation_strategy steps \
	        --max_steps 10000 \
	        --save_steps 5000 \
	        --eval_steps 500 \
	        --do_train \
	        --do_eval \
	        --optim adafactor \
	        --n_conversations 4 \
	        --instruction_prefix 'Rewrite the query based on the user-system conversation. query: {} conversation: ' \
	        --conversation_prefix 'user: {0} system: {1}' \
	        --warmup_steps 800 \
	        --report_to wandb
