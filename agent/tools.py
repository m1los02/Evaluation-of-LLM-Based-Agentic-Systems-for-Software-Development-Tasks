from typing import TypedDict
from . import sandbox

class ToolResult(TypedDict, total=False):
    observation: str
    artifact: dict

def run_python(code: str, tests: str, timeout_s: int = 3, mem_mb: int = 256) -> ToolResult:
    res = sandbox.execute(code, tests, timeout_s=timeout_s, mem_mb=mem_mb)
    if res["passed"]:
        return {"observation": "TESTS_PASS"}
    tb = (res.get("traceback") or "").strip()
    return {"observation": f"TESTS_FAIL\n{tb}"}

def inspect_traceback(traceback_text: str) -> ToolResult:
    if not traceback_text:
        return {"observation": "No traceback provided."}
    first_line = traceback_text.strip().splitlines()[0]
    return {"observation": f"Likely cause: {first_line}"}

def set_code(new_code: str) -> ToolResult:
    return {"observation": "Replaced entire file with new candidate code."}