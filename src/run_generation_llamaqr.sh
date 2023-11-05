torchrun --nproc_per_node 1 infer_llama2_qr.py \
    --ckpt_dir /tmp2/trec/pds/models/llama/llama-2-7b-chat/ \
    --tokenizer_path /tmp2/trec/pds/models/llama/tokenizer.model \
    --max_seq_len 1024 --max_batch_size 2 --filename prediction.jsonl 
