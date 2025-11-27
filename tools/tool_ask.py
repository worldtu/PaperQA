"""
Tool Ask - Parametric Knowledge Generator

Generates brief background knowledge (parametric knowledge) as supplement to retrieved context.
Based on PaperQA paper Appendix G.
"""

from typing import Optional
from openai import OpenAI


class AskTool:
    """
    Ask Tool - Generate Parametric Knowledge
    
    Generates 40-50 word background knowledge from LLM's pre-trained knowledge
    to supplement retrieved context and help detect contradictions.
    """
    
    def __init__(
        self,
        model: str = "llama3.1:8b",
        temperature: float = 0.3,
        max_words: int = 50
    ):
        """
        Initialize Ask Tool
        
        Args:
            model: Ollama model name
            temperature: Generation temperature (lower = more deterministic)
            max_words: Maximum words for background knowledge
        """
        self.model = model
        self.temperature = temperature
        self.max_words = max_words
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
    
    def generate_background(self, question: str) -> str:
        """
        Generate background knowledge (parametric knowledge) for the question
        
        Args:
            question: User question
            
        Returns:
            str: 40-50 word background knowledge
        """
        try:
            prompt = f"""You are a background knowledge provider. Your task is to provide ONLY background information about the topic, NOT to answer the question.

CRITICAL RULES:
- DO NOT answer the question directly
- DO NOT provide solutions, explanations, or conclusions
- DO provide established facts, key concepts, definitions, and general knowledge about the topic
- Focus on what is generally known about the subject matter
- Keep it concise (40-50 words) and factual

Question: {question}

Background Information Only (40-50 words, NO answer):"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=150,
            )
            
            background = response.choices[0].message.content.strip()
            background = self._truncate_to_words(background, self.max_words)
            return background
            
        except Exception:
            return ""
    
    def _truncate_to_words(self, text: str, max_words: int) -> str:
        """Truncate text to specified word count"""
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words])
