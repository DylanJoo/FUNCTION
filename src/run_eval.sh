grep=$1

rm result_${grep}.txt
for jsonl in results/qrecc_test/${grep}*jsonl;do
    echo ${jsonl##*/} >> result_${grep}.txt
    python3 evaluation.py \
        --jsonl $jsonl \
        --metric rouge \
        --metric bleu >> result_${grep}.txt
done
