"""
Answer LLM Tool
Generates final answers based on evidence and background.
"""

import ast
import json
from typing import List
from openai import OpenAI
from schemas import Chunk, Answer


class AnswerLLMTool:
    """Tool for generating final answers using LLM."""
    
    def __init__(self, model: str = "llama3.1:8b", temperature: float = 0.3):
        self.model = model
        self.temperature = temperature
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the Ollama client."""
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
    
    def answer(self, question: str, context_chunks: List[Chunk], 
              background: str) -> Answer:
        """
        Generate final answer based on context chunks and background.
        
        Args:
            question: Original question
            context_chunks: List of Chunk objects for context
            background: Background information (string)
            
        Returns:
            Answer object with answers, sources, and confidence
        """
        # Build context from chunks
        context = ""
        sources = []
        
        for i, chunk in enumerate(context_chunks[:8], 1):  # Top-8 chunks
            citation = f"[{i}] {chunk.paper_id}"
            context += f"\n\n{citation}\n{chunk.text}"
            sources.append(citation)
        
        answer_prompt = f"""You are a scientific question answering system. Based on the context provided, answer the question and return your response as a JSON object.

Context:
{context}

Background: {background}

Question: {question}

Your response MUST be a valid JSON object with exactly this structure:
{{
    "Answers": ["write your complete answer here with citation markers [1] [2] etc"],
    "Sources": [1, 2, 3],
    "Confidence": 0.85
}}

Rules:
- Use citation markers like [1], [2] to reference sources
- Sources should be a list of numbers (e.g., [1, 2, 3])
- Confidence is a number between 0.0 and 1.0
- If you cannot answer, put "I cannot answer" in Answers
- Return ONLY the JSON object above, no explanations before or after

JSON response:"""

        # Call Ollama API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": answer_prompt}],
                temperature=self.temperature,
                max_tokens=800,
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON if LLM added extra text
            if not content.startswith('{'):
                start_idx = content.find('{')
                end_idx = content.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    content = content[start_idx:end_idx+1]
                
        except Exception as e:
            return Answer(
                text="I cannot answer due to API error",
                citations=[],
                confidence=0.0,
                need_more=True
            )
        
        try:
            # Parse JSON response - try json.loads first, then ast.literal_eval as fallback
            try:
                content_json = json.loads(content)
            except (ValueError, json.JSONDecodeError):
                try:
                    content_json = ast.literal_eval(content)
                except Exception:
                    # If JSON parsing fails, return error answer
                    return Answer(
                        text="I cannot answer",
                        citations=[],
                        confidence=0.0,
                        need_more=True
                    )
            
            # Extract answer content
            answers_list = content_json.get("Answers", [])
            answer_text = " ".join(answers_list) if isinstance(answers_list, list) else str(answers_list)
            
            # Extract citations and confidence
            raw_sources = content_json.get("Sources", sources)
            answer_citations = []
            if isinstance(raw_sources, list):
                for src in raw_sources:
                    if isinstance(src, int):
                        if 1 <= src <= len(sources):
                            answer_citations.append(sources[src - 1])
                        else:
                            answer_citations.append(f"[{src}]")
                    else:
                        answer_citations.append(str(src))
            else:
                answer_citations = sources
            
            confidence = float(content_json.get("Confidence", 0.0))
            need_more = confidence < 0.7 or len(answer_citations) == 0
            
            return Answer(
                text=answer_text,
                citations=answer_citations,
                confidence=confidence,
                need_more=need_more
            )
            
        except Exception:
            return Answer(
                text="I cannot answer",
                citations=[],
                confidence=0.0,
                need_more=True
            )