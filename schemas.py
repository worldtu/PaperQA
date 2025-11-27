from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class SearchHit(BaseModel):
    source: str  # scholar|arxiv|pubmed|pmc|openaccess
    paper_id: str  # doi|arxiv:YYYY.NNNNN|pmid|pmcid
    title: str
    year: Optional[int] = None
    urls: Dict[str, str] = Field(default_factory=dict)  # {"pdf":..., "html":...}
    similarity: Optional[float] = None  # 用于存储语义相似度分数

class Document(BaseModel):
    paper_id: str
    sections: List[Dict] = Field(default_factory=list)  # [{"name":"Intro","start":0,"end":1234}]
    text: str

class Chunk(BaseModel):
    paper_id: str
    chunk_id: str
    start: int
    end: int
    text: str
    embedding: Optional[List[float]] = None

class ChunkRef(BaseModel):
    chunk_id: str
    paper_id: str
    rank: int
    scores: Dict[str, float]  # {"bm25":..., "vec":..., "mmr":...}

class Evidence(BaseModel):
    chunk_id: str
    summary: str
    score: Optional[float] = None  # 1-10 relevance

class Background(BaseModel):
    question: str
    background_text: str

class Answer(BaseModel):
    text: str
    citations: List[str]  # ["(Smith et al., 2022)"]
    confidence: float
    need_more: bool = False  # 是否需要更多信息
