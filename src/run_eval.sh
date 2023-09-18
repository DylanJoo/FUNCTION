# grep=$1
# 1) N-gram based evaluation
# rm result_${grep}.txt
# for jsonl in results/qrecc_test/${grep}*jsonl;do
#     echo ${jsonl##*/} >> result_${grep}.txt
#     python3 evaluation.py \
#         --jsonl $jsonl \
#         --metric rouge \
#         --metric bleu >> result_${grep}.txt
# done

qrels=dataset/scai_qrecc_test_qrel.trec
for run in runs/qrecc_test/*trec;do
    echo ${run##*/}
    trec_eval-9.0.7/trec_eval \
        -c -m recall.10,100 recip_rank ${qrels} ${run} |\
        cut -f3 | sed ':a; N; $!ba; s/\n/|/g'
done
