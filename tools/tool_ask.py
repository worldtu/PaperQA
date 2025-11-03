"""
PaperQA - tool_ask.py
Person C — Ask LLM (a priori background)

Thin adapter that wraps our AskLLMBackgroundEnhancer into the PaperQA tool API.
Output is capped to <= max_words (default 50).
"""

from dataclasses import asdict
from typing import Dict, Optional

from ask_llm_gemini import AskLLMBackgroundEnhancer, AskLLMConfig
from paperqa.schemas import Question, BackgroundKnowledge
from paperqa.settings import Settings


class ToolAsk:
    """Ask tool that generates priori background for a question."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings.from_env()
        self.enhancer = AskLLMBackgroundEnhancer(
            AskLLMConfig(
                model=self.settings.ask_model,
                temperature=self.settings.ask_temperature,
                max_tokens=self.settings.ask_max_tokens,
                max_words=self.settings.ask_max_words,
                cache_enabled=self.settings.cache_enabled,
                cache_ttl_hours=self.settings.cache_ttl_hours,
                retry_attempts=self.settings.max_retry_attempts,
                timeout_seconds=self.settings.timeout_seconds,
            ),
            api_key=self.settings.gemini_api_key,
        )

    def run(self, question: Question) -> BackgroundKnowledge:
        """Generate <= max_words background for the given question."""
        resp = self.enhancer.generate_background(question.text)
        return BackgroundKnowledge(
            question_id=question.id,
            background_text=resp.background_text,
            confidence_score=resp.confidence_score,
            generated_at=resp.generated_at,
            model_used=resp.model_used,
            tokens_used=resp.tokens_used,
            template_type=self._last_template(question.text),
        )

    # Helper to expose the template type used
    def _last_template(self, q: str) -> str:
        return self.enhancer._select_prompt_template(q)


def run_tool(question_dict: Dict) -> Dict:
    """Functional entry for simple pipelines.

    Args:
        question_dict: {"id": str, "text": str, "domain": Optional[str]}
    Returns:
        dict serialized BackgroundKnowledge
    """
    tool = ToolAsk()
    q = Question(**question_dict)
    bk = tool.run(q)
    return bk.to_dict()



