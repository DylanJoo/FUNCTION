# diff as reference
for file in results/ikat_train/*jsonl;do
    echo ${file##*/} >> results.csv
    python3 evaluation.py \
        --jsonl $file \
        --metric rouge \
        --metric bleu \
        --diff_as_reference >> results.csv
done
