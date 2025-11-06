import time
from typing import Dict
from agent.state import AgentState
from agent.llm import LLM
from agent.policy import step

def run_task(task, llm: LLM, max_steps: int = 6, timeout_s: int = 3, mem_mb: int = 256) -> Dict:
    state = AgentState(task_id=task.task_id, code=task.buggy_code, tests=task.tests, max_steps=max_steps)
    t0 = time.time()
    trace = []
    while not state.done and state.steps < state.max_steps:
        prev_code = state.code
        state = step(llm, state, timeout_s=timeout_s, mem_mb=mem_mb)
        trace.append({
            "step": state.steps,
            "last_assistant": next((m.content for m in reversed(state.messages) if m.role=="assistant"), None),
            "last_tool_obs":   next((m.content for m in reversed(state.messages) if m.role=="tool"), None),
            "code_head": state.code[:200],
            "code_changed": (state.code != prev_code),
        })
    t1 = time.time()
    return {
        "task_id": task.task_id,
        "passed": bool(state.passed),
        "steps": state.steps,
        "time_s": round(t1 - t0, 3),
        "trace": trace,
    }

