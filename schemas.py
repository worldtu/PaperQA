from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

@dataclass
class SearchHit:
    source: str                 # scholar|arxiv|pubmed|pmc|openaccess
    paper_id: str               # doi|arxiv:YYYY.NNNNN|pmid|pmcid
    title: str
    year: Optional[int] = None
    urls: Dict[str, str] = field(default_factory=dict)  # {"pdf": "...", "html": "..."}

@dataclass
class Chunk:
    paper_id: str
    chunk_id: str
    start: int
    end: int
    text: str

class SearchResult(BaseModel):
    """Search result from Person A (Search Tool) to Person B (Gather Evidence)."""
    paper_id: str = Field(..., description="Unique identifier for the paper")
    title: str = Field(..., description="Title of the paper")
    url: str = Field(..., description="URL to access the paper")
    year: str = Field(..., description="Publication year")
    chunk_texts: List[str] = Field(..., description="List of text chunks from the paper")

class Evidence(BaseModel):
    """Evidence from Person B (Gather Evidence) to Person D (Answer LLM)."""
    chunk_id: str = Field(..., description="Unique identifier for the text chunk")
    summary: str = Field(..., description="Summary of the chunk content")
    score: int = Field(..., ge=1, le=10, description="Relevance score from 1-10")
    citation: str               # "(Smith 2022)"

class Background(BaseModel):
    """Background information from Person C (Ask LLM) to Person D (Answer LLM)."""
    question: str = Field(..., description="Original question being answered")
    background_text: str = Field(..., description="Background information text")

class Citation(BaseModel):
    """Citation information for sources."""
    source: str = Field(..., description="Source file or paper")
    page: int = Field(..., description="Page number")
    content: Optional[str] = Field(None, description="Relevant content excerpt")

class Answer(BaseModel):
    """Final answer from Person D (Answer LLM)."""
    answers: List[str] = Field(..., description="List of selected answers")
    sources: List[str] = Field(..., description="List of source citations")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")

class Question(BaseModel):
    """Input question for the PaperQA system."""
    text: str = Field(..., description="The question text")
    question_id: Optional[str] = Field(None, description="Optional unique identifier")

class EvaluationResult(BaseModel):
    """Result of evaluating a single question."""
    question_id: str = Field(..., description="Unique identifier for the question")
    question: str = Field(..., description="The question text")
    predicted: str = Field(..., description="Predicted answer")
    correct: str = Field(..., description="Correct answer")
    is_correct: int = Field(..., description="1 if correct, 0 otherwise")
    is_unsure: int = Field(..., description="1 if unsure, 0 otherwise")
    is_incorrect: int = Field(..., description="1 if incorrect, 0 otherwise")

class EvaluationMetrics(BaseModel):
    """Aggregated evaluation metrics."""
    total_questions: int = Field(..., description="Total number of questions")
    correct: int = Field(..., description="Number of correct answers")
    incorrect: int = Field(..., description="Number of incorrect answers")
    unsure: int = Field(..., description="Number of unsure answers")
    accuracy: float = Field(..., description="Accuracy (correct/total)")
    precision: float = Field(..., description="Precision (correct/sure)")

class Config(BaseModel):
    """Configuration for different components."""
    answer_length: int = Field(default=500, description="Length of generated answers")
    summary_length: int = Field(default=500, description="Length of summaries")
    top_chunk_num: int = Field(default=3, description="Number of top chunks to use")
    k: int = Field(default=5, description="Number of search results")
    fetch_k: int = Field(default=20, description="Number of candidates for MMR")
    lambda_mult: float = Field(default=0.5, description="MMR diversity parameter")
    temperature: float = Field(default=0.6, description="LLM temperature")
    top_p: float = Field(default=0.95, description="LLM top-p parameter")
    top_k: int = Field(default=20, description="LLM top-k parameter")
    min_p: float = Field(default=0.0, description="LLM min-p parameter")
    presence_penalty: float = Field(default=1.5, description="LLM presence penalty")
