# paperqa/tools/tool_gather.py
from typing import List
from ..schemas import ChunkRef, Evidence, Chunk
from ..settings import EVIDENCE_TOPK_CONTEXT

class GatherTool:
    def __init__(self, retriever, summarizer):
        self.retriever = retriever  # has .mmr_search(question, k) -> List[Chunk]
        self.summarizer = summarizer  # LLM callable: (chunk, question) -> Evidence

    def gather(self, question: str) -> List[Evidence]:
        # 1) retrieve diverse chunks via MMR
        chunks: List[Chunk] = self.retriever.mmr_search(question, k=EVIDENCE_TOPK_CONTEXT)
        # 2) summarize concurrently (map step) with 1–10 score
        evidences: List[Evidence] = self._batch_summarize(chunks, question)
        # 3) sort by score and keep top-k in the context library
        evidences.sort(key=lambda e: e.score, reverse=True)
        return evidences

    def _batch_summarize(self, chunks: List[Chunk], question: str) -> List[Evidence]:
        # TODO: parallelize with ThreadPool/asyncio.gather
        return [self.summarizer(chunk=c, question=question) for c in chunks]