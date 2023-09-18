for rewritten in results/qrecc_test/*.jsonl; do 
    output_trec=${rewritten/results/runs}
    output_trec=${output_trec/jsonl/trec}
    mkdir -p ${output_trec%/*}
    python3 retrieve.py \
        --k 100 --k1 0.82 --b 0.68 \
        --batch_size 32 --threads 16 \
        --query ${rewritten} \
        --output ${output_trec} \
        --index /home/jhju/indexes/qrecc-commoncrawl-lucene/
done

# python3 retrieve.py \
#     --k 100 --k1 0.82 --b 0.68 \
#     --batch_size 32 --threads 16 \
#     --query results/qrecc_test/baseline_history_3-3.jsonl \
#     --output runs/manual.trec \
#     --use_manual \
#     --index /home/jhju/indexes/qrecc-commoncrawl-lucene/
