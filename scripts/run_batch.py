import argparse, json
from pathlib import Path
from evaluation.dataset import load_tasks
from evaluation.runners import run_task
from agent.llm import LLM
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--subset", type=int, default=None)
parser.add_argument("--out", required=True)
parser.add_argument("--config", default="configs/default.yaml")
parser.add_argument("--data_path", default=None, help="Path to HumanEvalFix JSONL (or similar)")
args = parser.parse_args()

cfg = yaml.safe_load(Path(args.config).read_text()) or {}
decode = cfg.get("decode", {}) or {}
debug = cfg.get("debug", {}) or {}

llm = LLM(cfg.get("model", "dummy"), **decode, **debug)

tasks = load_tasks(subset=args.subset, dataset_path=args.data_path)
total = len(tasks)
results = []
for i, t in enumerate(tasks, 1):
    print(f"[{i}/{total}] Running {t.task_id} ...", flush=True)
    res = run_task(t, llm, max_steps=cfg["agent"]["max_steps"], timeout_s=cfg["limits"]["timeout_s"])
    results.append(res)

Path(args.out).write_text("\n".join(json.dumps(r) for r in results), encoding="utf-8")
print(f"Wrote {len(results)} results to {args.out}")
