import argparse, json
from pathlib import Path
from evaluation.metrics import pass_at_1

parser = argparse.ArgumentParser()
parser.add_argument("--in", dest="inp", required=True)
args = parser.parse_args()

lines = Path(args.inp).read_text(encoding="utf-8").splitlines()
results = [json.loads(x) for x in lines if x.strip()]
print({"pass@1": pass_at_1(results), "n": len(results)})
