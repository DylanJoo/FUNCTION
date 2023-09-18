TRAIN_FILE=/home/jhju/datasets/qrecc/qrecc_train.json
EVAL_FILE=/home/jhju/datasets/qrecc/qrecc_test.json
TEST_FILE=dataset/2023_test_topics.json
BASE=google/flan-t5-base

preprocess:
	# convert naacl baseline to run
	python3 utils/convert_scai_baseline_to_run.py \
	        --scai-baseline-json dataset/scai-qrecc21-naacl-baseline.json
	# convert qrels to trec
	python3 utils/convert_scai_qrels_to_trec.py \
	        --scai-qrels-json dataset/scai_qrecc_test_qrel.json 

# 22 G
train_flatten:
	python3 train_flatten.py \
     		--model_name_or_path ${BASE} \
     		--tokenizer_name ${BASE} \
		--config_name ${BASE} \
     		--train_file ${TRAIN_FILE} \
     		--eval_file ${EVAL_FILE} \
		--output_dir models/ckpt/function-base-flatten \
	        --per_device_train_batch_size 8 \
	        --max_src_length 256 \
	        --max_tgt_length 64 \
	        --learning_rate 1e-4 \
	        --evaluation_strategy steps \
	        --max_steps 20000 \
	        --save_steps 5000 \
	        --eval_steps 500 \
	        --do_train \
	        --do_eval \
	        --optim adafactor \
	        --n_conversations 6 \
	        --warmup_steps 1000 \
		--lr_scheduler_type linear \
		--instruction_prefix 'Based on previous conversations, rewrite the user utterance: {} into a standalone query.' \
		--conversation_prefix 'user: {0} system: {1}' \
	        --report_to wandb

train_ntr:
	python3 train_ntr.py \
     		--model_name_or_path ${BASE} \
     		--tokenizer_name ${BASE} \
		--config_name ${BASE} \
     		--train_file ${TRAIN_FILE} \
     		--eval_file ${EVAL_FILE} \
		--output_dir models/ckpt/ntr-base \
	        --per_device_train_batch_size 8 \
	        --max_src_length 512 \
	        --max_tgt_length 64 \
	        --learning_rate 1e-3 \
	        --evaluation_strategy steps \
	        --max_steps 20000 \
	        --save_steps 5000 \
	        --eval_steps 500 \
	        --do_train \
	        --do_eval \
	        --optim adafactor \
	        --n_conversations 6 \
	        --warmup_steps 0 \
	        --lr_scheduler_type linear \
	        --report_to wandb

# 17 G
train_compressed:
	export CUDA_VISIBLE_DEVICES=0; python3 train_compressed.py \
     		--model_name_or_path ${BASE} \
     		--tokenizer_name ${BASE} \
		--config_name ${BASE} \
     		--train_file ${TRAIN_FILE} \
     		--eval_file ${EVAL_FILE} \
		--output_dir models/ckpt/function-base-compressed \
	        --per_device_train_batch_size 16 \
	        --max_src_length 32 \
	        --max_tgt_length 64 \
	        --max_src_conv_length 128 \
	        --learning_rate 1e-4 \
	        --evaluation_strategy steps \
	        --max_steps 10000 \
	        --save_steps 5000 \
	        --eval_steps 500 \
	        --do_train \
	        --do_eval \
	        --optim adafactor \
	        --n_conversations 10 \
	        --warmup_steps 0 \
		--instruction_prefix 'Rewrite the user utterance based on previous user-system conversations. user: {}' \
		--conversation_prefix 'conversation: user: {0} system: {1}' \
	        --lr_scheduler_type linear \
	        --report_to wandb

rewrite_by_t5ntr:
	python3 generate_ikat.py \
		--model_name castorini/t5-base-canard \
		--model_path castorini/t5-base-canard \
		--input_file ${TEST_FILE} \
		--output_jsonl results/ikat_test/t5ntr_history_3-3.jsonl \
		--device cuda:0 \
		--batch_size 4 \
		--n_conversations 3 \
		--n_responses 3 \
		--num_beams 5 \
		--max_src_length 512 \
		--max_tgt_length 256

index_bm25:
	python3 -m pyserini.index.lucene \
	  --collection JsonCollection \
	  --input /home/jhju/datasets/qrecc/collection-paragraph/ \
	  --index /home/jhju/indexes/qrecc-commoncrawl-lucene/  \
	  --generator DefaultLuceneDocumentGenerator \
	  --threads 8
