"""
Shared schemas (interfaces) for PaperQA tools.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Question:
    id: str
    text: str
    domain: Optional[str] = None
    created_at: datetime = datetime.now()


@dataclass
class BackgroundKnowledge:
    question_id: str
    background_text: str
    confidence_score: float
    generated_at: datetime
    model_used: str
    tokens_used: int
    template_type: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "background_text": self.background_text,
            "confidence_score": self.confidence_score,
            "generated_at": self.generated_at.isoformat(),
            "model_used": self.model_used,
            "tokens_used": self.tokens_used,
            "template_type": self.template_type,
        }

# ===== Types used by search/gather tools (light dataclasses) =====

@dataclass
class SearchHit:
    source: str
    paper_id: str
    title: str
    year: Optional[int]
    urls: Dict[str, str]


@dataclass
class Chunk:
    paper_id: str
    chunk_id: str
    start: int
    end: int
    text: str
    embedding: Optional[List[float]] = None


@dataclass
class Evidence:
    chunk_id: str
    summary: str
    score: Optional[float]

