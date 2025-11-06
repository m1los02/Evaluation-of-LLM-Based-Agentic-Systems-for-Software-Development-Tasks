import runpy
import types
import threading
import io
import sys
from contextlib import redirect_stdout, redirect_stderr

class RunResult(dict):
    pass

def _exec_with_timeout(func, timeout_s: int):
    th = threading.Thread(target=func)
    th.daemon = True
    th.start()
    th.join(timeout=timeout_s)
    if th.is_alive():
        raise TimeoutError("Sandbox timeout")

def execute(code: str, tests: str, timeout_s: int = 3, mem_mb: int = 256) -> RunResult:
    
    stdout, stderr = io.StringIO(), io.StringIO()
    trace = None
    passed = False

    def run_all():
        nonlocal passed, trace
        code_globals = {"__name__": "code"}
        exec(code, code_globals)
        test_globals = {**code_globals}
        exec(tests, test_globals)
        had_test = False
        for k, v in list(test_globals.items()):
            if k.startswith("test_") and callable(v):
                had_test = True
                v()
        passed = True

    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            _exec_with_timeout(run_all, timeout_s)
    except TimeoutError:
        trace = "TIMEOUT"
    except Exception as e:
        trace = f"{type(e).__name__}: {e}"

    return RunResult(
        passed=passed,
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
        traceback=trace,
    )
