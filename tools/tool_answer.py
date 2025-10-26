"""AnswerTool implementation following PaperQA paper specification.
- Ask LLM: Provides background information from pre-trained LLM
- Answer LLM: Generates final answer with citations
- Follows exact prompt format from the paper
"""

from dataclasses import dataclass
from typing import List, Optional
import re

# Schemas matching the project structure
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

# Ask LLM prompt (from paper)
ASK_PROMPT = "Do you know anything about this question?"

# Answer LLM prompt (exact format from paper)
ANSWER_PROMPT = """Write an answer (answer_length={answer_length}) for the question below based on the provided context. If the context provides insufficient information, reply ''I cannot answer''. For each part of your answer, indicate which sources most support it via valid citation markers at the end of sentences, like (Vaswani2017) or (Devlin2018). Answer in an unbiased, comprehensive, and scholarly tone. If the question is subjective, provide an opinionated answer in the concluding 1-2 sentences.

IMPORTANT: You MUST include citations in the format (AuthorYYYY) for each claim you make. Use the citations provided in the context.

Context:
{context}

Extra background information: {background}

Question: {question}

Answer:"""

class AnswerTool:
    """AnswerTool following PaperQA paper specification.
    
    Implements the two-step process:
    1. Ask LLM: Get background information from pre-trained LLM
    2. Answer LLM: Generate final answer with citations
    """

    def __init__(self, ask_llm=None, answer_llm=None, answer_length: str = "medium", 
                 min_evidence: int = 1, top_k: int = 8, min_avg_score: float = 6.5):
        self.ask_llm = ask_llm
        self.answer_llm = answer_llm
        self.answer_length = answer_length
        self.min_evidence = min_evidence
        self.top_k = top_k
        self.min_avg_score = min_avg_score

    def _get_ask_llm_response(self, question: str) -> str:
        """Get background information from ask LLM."""
        if self.ask_llm is None:
            # Fallback: return empty background
            return ""
        
        try:
            # Call ask LLM with the question
            response = self.ask_llm.generate(
                prompt=ASK_PROMPT,
                question=question,
                max_tokens=100,
                temperature=0.1
            )
            return response.strip()
        except Exception as e:
            print(f"Ask LLM error: {e}")
            return ""

    def _get_answer_llm_response(self, question: str, context: str, background: str) -> str:
        """Get final answer from answer LLM."""
        if self.answer_llm is None:
            # Fallback: return simple response
            return "I cannot answer."
        
        try:
            # Format the prompt according to paper specification
            prompt = ANSWER_PROMPT.format(
                answer_length=self.answer_length,
                context=context,
                background=background,
                question=question
            )
            
            response = self.answer_llm.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.3
            )
            return response.strip()
        except Exception as e:
            print(f"Answer LLM error: {e}")
            return "I cannot answer."

    def _extract_citations(self, text: str) -> List[str]:
        """Extract citations from answer text."""
        # Find citations in format (AuthorYYYY) - more flexible pattern
        citation_patterns = [
            r'\([A-Za-z]+\d{4}\)',  # (AuthorYYYY)
            r'\([A-Za-z]+\s+\d{4}\)',  # (Author YYYY)
            r'\[[A-Za-z]+\d{4}\]',  # [AuthorYYYY]
            r'\[[A-Za-z]+\s+\d{4}\]',  # [Author YYYY]
        ]
        
        citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)
        
        # Clean up citations and remove duplicates
        cleaned_citations = []
        for citation in citations:
            # Remove brackets/parentheses and clean up
            clean_citation = citation.strip('()[]')
            if clean_citation and clean_citation not in cleaned_citations:
                cleaned_citations.append(f"({clean_citation})")
        
        return cleaned_citations

    def generate(self, question: str, evidences: List[Evidence], background: Optional[Background] = None) -> Answer:
        """Generate answer following PaperQA specification."""
        
        # Step 1: Check if we have sufficient evidence
        if not evidences:
            return Answer(text="I cannot answer.", citations=[], confidence=0.0)

        # Sort by score and take top_k
        ev = sorted(evidences, key=lambda e: e.score, reverse=True)[:self.top_k]
        avg_score = sum(e.score for e in ev) / len(ev)

        # Check evidence quality
        if len(ev) < self.min_evidence or avg_score < self.min_avg_score:
            return Answer(text="I cannot answer.", citations=[], confidence=0.0)

        # Step 2: Get background from ask LLM
        ask_background = self._get_ask_llm_response(question)
        
        # Combine with provided background if available
        if background and background.background_text:
            ask_background = f"{ask_background}\n{background.background_text}".strip()

        # Step 3: Build context from evidence with clear citation format
        context_parts = []
        for i, e in enumerate(ev, 1):
            context_parts.append(f"Source {i}: {e.summary} {e.citation}")
        context = "\n".join(context_parts)

        # Step 4: Get answer from answer LLM
        answer_text = self._get_answer_llm_response(question, context, ask_background)
        
        # Step 5: Extract citations and calculate confidence
        citations = self._extract_citations(answer_text)
        confidence = min(0.99, max(0.0, avg_score / 10.0))

        return Answer(text=answer_text, citations=citations, confidence=confidence)
