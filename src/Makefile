TRAIN_FILE=/home/jhju/datasets/qrecc/qrecc_train.json
EVAL_FILE=/home/jhju/datasets/qrecc/qrecc_test.json

# 22 G
train_flatten:
	export CUDA_VISIBLE_DEVICES=2; python3 train_flatten.py \
     		--model_name_or_path google/flan-t5-base \
     		--tokenizer_name google/flan-t5-base \
		--config_name google/flan-t5-base \
     		--train_file ${TRAIN_FILE} \
     		--eval_file ${EVAL_FILE} \
		--output_dir models/ckpt/function-base-flatten \
	        --per_device_train_batch_size 8 \
	        --max_src_length 128 \
	        --max_tgt_length 32 \
	        --learning_rate 1e-4 \
	        --evaluation_strategy steps \
	        --max_steps 20000 \
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

# 17 G
train_compressed:
	export CUDA_VISIBLE_DEVICES=2
	python3 train_compressed.py \
     		--model_name_or_path google/flan-t5-base \
     		--tokenizer_name google/flan-t5-base \
		--config_name google/flan-t5-base \
     		--train_file ${TRAIN_FILE} \
     		--eval_file ${EVAL_FILE} \
		--output_dir models/ckpt/function-base-compressed \
	        --per_device_train_batch_size 8 \
	        --max_src_length 32 \
	        --max_tgt_length 32 \
	        --max_src_conv_length 160 \
	        --learning_rate 1e-4 \
	        --evaluation_strategy steps \
	        --max_steps 20000 \
	        --save_steps 5000 \
	        --eval_steps 500 \
	        --do_train \
	        --do_eval \
	        --optim adafactor \
	        --n_conversations 6 \
	        --instruction_prefix 'Rewrite the query based on the user-system conversation. query: {} conversation: ' \
	        --conversation_prefix 'user: {0} system: {1}' \
	        --warmup_steps 800 \
	        --report_to wandb

generate_by_function_flat:
	python3 generate.py \
		--model_name google/flan-t5-base \
		--model_path models/ckpt/function-base-flatten/checkpoint-20000 \
		--input_file ${EVAL_FILE} \
		--output_jsonl results/qrecc_test.function_flat.prediction.jsonl \
		--device cuda:1 \
		--batch_size 4 \
		--instruction_prefix 'Rewrite the query based on the user-system conversation. query: {} conversation: ' \
		--conversation_prefix 'user: {0} system: {1}' \
		--n_conversations 8 \
		--num_beams 5 \
		--max_src_length 256 \
		--max_tgt_length 32

generate_by_function_comp:
	export CUDA_VISIBLE_DEVICES=1; python3 generate.py \
		--model_name google/flan-t5-base \
		--model_path models/ckpt/function-base-compressed/checkpoint-20000 \
		--input_file ${EVAL_FILE} \
		--output_jsonl results/qrecc_test.function_comp.prediction.jsonl \
		--device cuda \
		--batch_size 4 \
		--instruction_prefix 'Rewrite the query based on the user-system conversation. query: {} conversation: ' \
		--conversation_prefix 'user: {0} system: {1}' \
		--n_conversations 8 \
		--num_beams 5 \
		--max_src_length 64 \
		--max_tgt_length 32 \
		--max_src_conv_length 256

generate_by_t5ntr:
	python3 generate.py \
		--model_name castorini/t5-base-canard \
		--model_path castorini/t5-base-canard \
		--input_file ${EVAL_FILE} \
		--output_jsonl results/qrecc_test.ntr.prediction.jsonl \
		--device cuda \
		--batch_size 4 \
		--n_conversations 3 \
		--num_beams 5 \
		--max_src_length 512 \
		--max_tgt_length 32
