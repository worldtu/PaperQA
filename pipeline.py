"""
PaperQA pipeline: wires tool_search, tool_gather, tool_ask, tool_answer.

Assumptions:
- Each tool exposes a `run_tool(input_dict) -> dict` function.
- Root-level modules: tool_search.py, tool_gather.py, tool_answer.py
- Ask tool from paperqa.tools.tool_ask.run_tool

This module provides a simple synchronous end-to-end flow for a single question.
"""

from typing import Dict, Any
import importlib
import time

from paperqa.tools.tool_ask import run_tool as run_tool_ask


class Pipeline:
    def __init__(self):
        # dynamic import root-level tools
        self.tool_search = importlib.import_module("tool_search")
        self.tool_gather = importlib.import_module("tool_gather")
        self.tool_answer = importlib.import_module("tool_answer")

    def run(self, question_text: str, question_id: str = "q_demo") -> Dict[str, Any]:
        t0 = time.time()

        # Step C: Ask LLM (a priori background)
        bk = run_tool_ask({"id": question_id, "text": question_text})

        # Step A: Search
        search_in = {"question": question_text}
        search_out = self.tool_search.run_tool(search_in)

        # Step B: Gather Evidence (use background + search results)
        gather_in = {
            "question": question_text,
            "search_result": search_out,
            "background": bk,
        }
        gather_out = self.tool_gather.run_tool(gather_in)

        # Step D: Answer (use evidence + background)
        answer_in = {
            "question": question_text,
            "evidence": gather_out,
            "background": bk,
        }
        answer_out = self.tool_answer.run_tool(answer_in)

        return {
            "question": question_text,
            "background": bk,
            "search": search_out,
            "evidence": gather_out,
            "answer": answer_out,
            "latency_s": round(time.time() - t0, 2),
        }



