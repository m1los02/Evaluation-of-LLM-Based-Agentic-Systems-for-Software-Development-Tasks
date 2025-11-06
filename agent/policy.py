import re
from .state import AgentState, AgentMessage
from .llm import LLM
from . import tools

SYSTEM = (
    "You are a code-fixing assistant.\n"
    "Respond with EXACTLY ONE directive and nothing else:\n"
    "- TOOL: run_python\n"
    "- TOOL: inspect_traceback\n"
    "- TOOL: set_code\n<code>\n<ENTIRE CORRECTED FILE HERE>\n</code>\n"
    "- FINAL_ANSWER: <short status>\n"
    "Do NOT include <think> or any prose outside the directive.\n"
    "Examples:\n"
    "TOOL: set_code\n<code>\n"
    "def take_first_n(arr, n):\n"
    "    return arr[:n+1]\n"
    "</code>\n\n"
    "TOOL: set_code\n<code>\n"
    "def add(a, b):\n"
    "    return a + b\n"
    "</code>\n"
)

def decide_next_action(llm: LLM, state: AgentState) -> str:
    msgs = [{"role": "system", "content": SYSTEM}]
    msgs += [{"role": m.role, "content": m.content} for m in state.messages]
    context = f"CODE:\n{state.code}\n\nTESTS (head):\n{state.tests[:400]}"
    msgs.append({"role": "user", "content": context})
    return llm.chat(msgs)

def step(llm: LLM, state: AgentState, timeout_s: int = 3, mem_mb: int = 256) -> AgentState:
    action = decide_next_action(llm, state)
    state.steps += 1

    if action.startswith("FINAL_ANSWER:"):
        state.done = True
        state.passed = ("All tests pass" in action) or ("TESTS_PASS" in action)
        state.messages.append(AgentMessage(role="assistant", content=action))
        return state

    if action.startswith("TOOL: set_code"):
        body = action.split("TOOL: set_code", 1)[1].replace("\r\n", "\n")
        code_body = ""

        m = re.search(r"(?is)<code>\n?(?P<body>.*)\n?</code>", body)
        if m:
            code_body = m.group("body").strip()
        else:
            # also accept fenced blocks
            m2 = re.search(r"(?is)```(?:python)?\n(?P<body>.*?)\n```", body)
            if m2:
                code_body = m2.group("body").strip()
            else:
                code_body = body.strip()

        obs = tools.set_code(code_body)
        state.code = code_body
        state.messages.append(AgentMessage(role="assistant", content=action))
        state.messages.append(AgentMessage(role="tool", content=obs["observation"]))

        res = tools.run_python(state.code, state.tests, timeout_s=timeout_s, mem_mb=mem_mb)
        state.messages.append(AgentMessage(role="assistant", content="TOOL: run_python"))
        state.messages.append(AgentMessage(role="tool", content=res["observation"]))
        if res["observation"].startswith("TESTS_PASS"):
            state.done = True
            state.passed = True
        elif state.steps >= state.max_steps:
            state.done = True
            state.passed = False
        return state

    if action.strip() == "TOOL: run_python":
        res = tools.run_python(state.code, state.tests, timeout_s=timeout_s, mem_mb=mem_mb)
        obs = res["observation"]
        state.messages.append(AgentMessage(role="assistant", content=action))
        state.messages.append(AgentMessage(role="tool", content=obs))
        if obs.startswith("TESTS_PASS"):
            state.done = True
            state.passed = True
        elif state.steps >= state.max_steps:
            state.done = True
            state.passed = False
        return state

    if action.strip() == "TOOL: inspect_traceback":
        tb = ""
        for m in reversed(state.messages):
            if m.role == "tool" and m.content.startswith("TESTS_FAIL"):
                tb = m.content.split("\n", 1)[1] if "\n" in m.content else ""
                break
        res = tools.inspect_traceback(tb)
        state.messages.append(AgentMessage(role="assistant", content=action))
        state.messages.append(AgentMessage(role="tool", content=res["observation"]))
        return state

    state.messages.append(AgentMessage(role="assistant", content=action))
    state.done = True
    state.passed = False
    return state
