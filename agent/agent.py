# paperqa/agent/agent.py
from ..settings import EVIDENCE_MIN, MAX_AGENT_STEPS
from .policy import AgentState, Policy

class PaperQAgent:
    def __init__(self, registry):
        self.registry = registry
        self.policy = Policy()

    def run(self, question: str):
        state = AgentState(question=question)
        # 1) initial search (paper requires a first search on full question)
        hits = self.registry.search.search(question)
        # optionally: ingest/parse/chunk/index if A owns parsing; otherwise B produces chunks offline

        while not self.policy.should_answer(state, EVIDENCE_MIN, MAX_AGENT_STEPS):
            state.tries += 1
            # 2) gather evidence (map step: MMR + summaries scored 1–10)
            round_evidence = self.registry.gather.gather(question)
            # Merge new evidence by chunk_id (dedupe)
            known = {e.chunk_id for e in state.evidence}
            for e in round_evidence:
                if e.chunk_id not in known:
                    state.evidence.append(e)

            # If still weak evidence, policy may switch to re-search (not shown: keyword reformulation)
            # For a minimal version, you can keep gathering a couple of times before answering.
