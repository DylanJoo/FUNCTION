import json
import argparse

def convert_scai_qrels_to_trec(path):
    fout = open(path.replace('json', 'trec'), 'w')
    truths = json.load(open(path, "r"))
    for truth in truths:
        qid = f"{truth['Conversation_no']}_{truth['Turn_no']}"
        for p in truth['Truth_passages']:
            fout.write(f"{qid}\t0\t{p}\t1\n")
    fout.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # load model
    parser.add_argument("--scai-qrels-json", type=str)
    args = parser.parse_args()

    convert_scai_qrels_to_trec(args.scai_qrels_json)
