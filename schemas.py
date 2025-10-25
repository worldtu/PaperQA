from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class SearchHit:
    source: str                 # scholar|arxiv|pubmed|pmc|openaccess
    paper_id: str               # doi|arxiv:YYYY.NNNNN|pmid|pmcid
    title: str
    year: Optional[int] = None
    urls: Dict[str, str] = field(default_factory=dict)  # {"pdf": "...", "html": "..."}

@dataclass
class Document:
    paper_id: str
    sections: List[Dict] = field(default_factory=list)  # [{"name": "Intro", "start": 0, "end": 1234}]
    text: str = ""

@dataclass
class Chunk:
    paper_id: str
    chunk_id: str
    start: int
    end: int
    text: str

@dataclass
class ChunkRef:
    chunk_id: str
    paper_id: str
    rank: int
    scores: Dict[str, float]    # {"bm25": ..., "vec": ..., "mmr": ...}

@dataclass
class Evidence:
    chunk_id: str
    summary: str
    score: float                # 1–10
    citation: str               # "(Smith 2022)"

@dataclass
class Background:
    question: str
    background_text: str

@dataclass
class Answer:
    text: str
    citations: List[str]
    confidence: float
