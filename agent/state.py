from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class AgentMessage:
    role: str  # one of the: "system", "user", "assistant", "tool"
    content: str

@dataclass
class AgentState:
    task_id: str
    code: str
    tests: str
    messages: List[AgentMessage] = field(default_factory=list)
    steps: int = 0
    max_steps: int = 3
    done: bool = False
    passed: Optional[bool] = None
    last_traceback: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

