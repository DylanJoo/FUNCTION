import json
import argparse

def convert_scai_json_to_run(path):
    path = path.replace('dataset', 'runs/qrecc_test')
    path = path.replace('json', 'trec')
    fout = open(path, 'w')
    baselines = json.load(open(path, "r"))
    for baseline in baselines:
        qid = f"{baseline['Conversation_no']}_{baseline['Turn_no']}"
        if 'Model_passages' in baseline.keys():
            for i, (p, s) in enumerate(baseline['Model_passages'].items()):
                fout.write(f"{qid} Q0 {p} {i+1} {s} baseline\n")
    fout.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # load model
    parser.add_argument("--scai-baseline-json", type=str)
    args = parser.parse_args()

    convert_scai_json_to_run(args.scai_baseline_json)
