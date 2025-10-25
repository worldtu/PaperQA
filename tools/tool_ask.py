# paperqa/tools/tool_ask.py
from ..schemas import Background

ASK_PROMPT = (
    "We are collecting background information for the question below. "
    "Provide ~50 words that could help answer the question. Do not answer directly."
)

class AskTool:
    def __init__(self, llm, cache):
        self.llm = llm
        self.cache = cache  # memoize per-question

    def ask(self, question: str) -> Background:
        if question in self.cache:
            return self.cache[question]
        # TODO: call local LLM, low temp, short max tokens
        text = ""
        bg = Background(question=question, background_text=text)
        self.cache[question] = bg
        return bg