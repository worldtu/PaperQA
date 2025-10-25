# paperqa/agent/policy.py
from dataclasses import dataclass, field
from typing import List

@dataclass
class AgentState:
    question: str
    tries: int = 0
    evidence: List = field(default_factory=list)

class Policy:
    def should_answer(self, state: AgentState, evidence_min: int, max_steps: int) -> bool:
        return len(state.evidence) >= evidence_min or state.tries >= max_steps

    def next_action(self, state: AgentState) -> str:
        # simple policy: prioritize gather; if gather yielded too little, trigger re-search
        if state.tries == 0:
            return "search"
        # If last gather added little evidence, try re-search; else keep gathering once more
        return "gather"