from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import json
import re

@dataclass
class Task:
    task_id: str
    buggy_code: str
    tests: str

def _norm_newlines(s: str) -> str:
    return s.replace("\r\n", "\n")

def _pick_first_nonempty(*candidates: Optional[str]) -> Optional[str]:
    for c in candidates:
        if c is not None and str(c).strip():
            return str(c)
    return None

def _entry_point_to_stub(entry_point: str, prompt: Optional[str]) -> str:
    header = f"def {entry_point}(*args, **kwargs):\n    pass\n"
    if prompt:
        m = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*:", prompt)
        if m:
            # keep signature but body pass
            sig = prompt[prompt.find("def"):prompt.find(":")+1]
            header = sig + "\n    pass\n"
    return header

def load_humaneval_jsonl(path: str, limit: Optional[int] = None) -> List[Task]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    tasks: List[Task] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)

            tid = _pick_first_nonempty(obj.get("task_id"), obj.get("id"))
            tests = _pick_first_nonempty(obj.get("tests"), obj.get("test"))
            buggy = _pick_first_nonempty(obj.get("buggy_code"), obj.get("buggy_solution"), obj.get("buggy"))

            if buggy is None:
                prompt = obj.get("prompt")
                entry = obj.get("entry_point")
                if entry:
                    buggy = _entry_point_to_stub(entry, prompt)

            if not tid or not tests or not buggy:
                continue

            tasks.append(
                Task(
                    task_id=str(tid),
                    buggy_code=_norm_newlines(buggy),
                    tests=_norm_newlines(tests),
                )
            )
            if limit is not None and len(tasks) >= limit:
                break
    return tasks

def load_tasks(
    subset: int | None = None,
    seed: int | None = None,
    dataset_path: Optional[str] = None
) -> List[Task]:
    if dataset_path:
        return load_humaneval_jsonl(dataset_path, limit=subset)