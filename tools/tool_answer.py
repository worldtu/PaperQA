"""Super-simple AnswerTool aligned with the PaperQA distribution (Person D).
- Map: handled by GatherTool upstream (inputs here are Evidence[]).
- Ask (a priori): Background injected as plain text.
- Reduce: Compose a cited answer or say "I cannot answer".
Keep it tiny, deterministic, and dependency-free.
"""

from dataclasses import dataclass
from typing import List

# Minimal schemas (standalone). Swap to your project's schemas if available.
@dataclass
class Evidence:
    chunk_id: str
    summary: str
    score: float          # 1–10 relevance
    citation: str         # e.g., "(Smith 2022)"

@dataclass
class Background:
    question: str
    background_text: str  # ~50 words

@dataclass
class Answer:
    text: str
    citations: List[str]
    confidence: float     # 0–1
    
ANSWER_PROMPT = (
    "Write an answer (answer_length={answer_length}) for the question based on the provided context. "
    "If insufficient, reply 'I cannot answer'. Cite sources as (AuthorYYYY). "
    "The answer should be in the following format: "
    "Answer: <answer> "
    "Citations: <citations> "
    "Confidence: <confidence> "
    "Answer length: <answer_length> "
    "Citations: <citations> "
    "Confidence: <confidence> "
    "Answer length: <answer_length> "
)

class AnswerTool:
    """Reduce step: synthesize final answer from evidence + background.
    Rules (paper-faithful, simplified):
      - Use up to top_k evidence summaries (default 8).
      - If not enough evidence OR low scores, return "I cannot answer".
      - Otherwise write a concise, cited answer.
    """

    def __init__(self, llm, min_evidence: int = 5, top_k: int = 8, min_avg_score: float = 6.5):
        self.llm = llm
        self.min_evidence = min_evidence
        self.top_k = top_k
        self.min_avg_score = min_avg_score

    def generate(self, question: str, evidences: List[Evidence], background: Background) -> Answer:
        if not evidences:
            return Answer(text="I cannot answer.", citations=[], confidence=0.0)

        # sort by score desc and take top_k
        ev = sorted(evidences, key=lambda e: e.score, reverse=True)[: self.top_k]
        avg_score = sum(e.score for e in ev) / len(ev)

        # Insufficient evidence guardrails (paper requires "I cannot answer" fallback)
        if len(ev) < self.min_evidence or avg_score < self.min_avg_score:
            return Answer(text="I cannot answer.", citations=[], confidence=0.0)

        # Compose a compact answer: background (optional) + 2–3 fused points from evidence
        bullets = []
        for e in ev[:3]:  # keep it short
            bullets.append(f"- {e.summary.strip()} {e.citation}")

        bg = background.background_text.strip()
        parts = []
        if bg:
            parts.append(f"Background: {bg}")
        parts.append("Key points:\n" + "\n".join(bullets))
        parts.append("Conclusion: Based on the retrieved evidence above, this is the best-supported answer.")
        text = "\n\n".join(parts)

        # Citations: unique order-preserving
        seen = set()
        citations = []
        for e in ev:
            if e.citation and e.citation not in seen:
                citations.append(e.citation)
                seen.add(e.citation)

        # Confidence: normalized mean relevance (simple, transparent)
        confidence = min(0.99, max(0.0, sum(e.score for e in ev) / (10.0 * len(ev))))

        return Answer(text=text, citations=citations, confidence=confidence)
