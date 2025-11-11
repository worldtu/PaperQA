"""
PaperQA — Gather Evidence Tool
Implements: MMR retrieval + Summary LLM + LLM-based scoring (1–10 relevance)
"""
import os
import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*tokenizers.*")

from concurrent.futures import ThreadPoolExecutor
import re
import json
import random
import asyncio
from typing import List, Optional
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# 加载 .env 文件（如果存在）
load_dotenv()
from paperqa.schemas import Chunk, Evidence
from paperqa.settings import (
    EVIDENCE_TOPK_CONTEXT,
    MMR_LAMBDA,
    MMR_INITIAL_TOPK,
)
# ----------------------------------------------------
# Retriever: use MMR to select diverse & relevant chunks
# ----------------------------------------------------
class Retriever:
    def __init__(self, chunks: List[Chunk], lambda_param: float = MMR_LAMBDA, initial_topk: int = MMR_INITIAL_TOPK):
        self.chunks = chunks
        self.lambda_param = lambda_param
        self.initial_topk = initial_topk  # 先做粗排，召回 top-N 个最相关的
        
    def mmr_search(self, query_emb: np.ndarray, k: int = EVIDENCE_TOPK_CONTEXT) -> List[Chunk]:
        """
        Return top-k diverse and relevant chunks using Maximal Marginal Relevance (MMR).
        
        改进：两阶段检索
        1. 粗排：先用相似度召回 top-N 个最相关的 chunks
        2. 精排：在 top-N 上用 MMR 选择既相关又多样化的 k 个 chunks
        """
        # Check that all chunks have embeddings (should be set in search.ingest())
        chunks_without_embedding = [c for c in self.chunks if c.embedding is None]
        if chunks_without_embedding:
            raise ValueError(f"{len(chunks_without_embedding)} chunks missing embeddings. "
                           "Embeddings should be generated in SearchTool.ingest().")
        
        # Flatten query_emb if it's 2D
        if len(query_emb.shape) > 1:
            query_emb = query_emb.flatten()
        
        # 计算所有 chunks 与查询的相似度
        docs = np.stack([np.array(c.embedding) for c in self.chunks])
        sim_to_query = np.dot(docs, query_emb) / (
            np.linalg.norm(docs, axis=1) * np.linalg.norm(query_emb) + 1e-8  # 防止除零
        )
        
        # 🔍 阶段1：粗排 - 先召回 top-N 个最相关的 chunks
        # 确保 initial_topk 不超过总 chunks 数
        actual_topk = min(self.initial_topk, len(self.chunks))
        top_indices = np.argsort(sim_to_query)[::-1][:actual_topk]  # 按相似度降序排列
        
        # 只在 top-N 候选集上计算相似度矩阵（减少计算量）
        candidate_docs = docs[top_indices]
        candidate_sim_to_query = sim_to_query[top_indices]
        candidate_sim_matrix = np.dot(candidate_docs, candidate_docs.T)
        
        # 🔍 阶段2：精排 - 在 top-N 上用 MMR 选择 k 个
        selected, remaining = [], list(range(len(top_indices)))

        while len(selected) < k and remaining:
            mmr_scores = []
            for i in remaining:
                # 与查询的相关性
                relevance = candidate_sim_to_query[i]
                # 与已选文档的最大冗余度
                redundancy = max([candidate_sim_matrix[i, j] for j in selected], default=0)
                # MMR 分数：平衡相关性和多样性
                score = self.lambda_param * relevance - (1 - self.lambda_param) * redundancy
                mmr_scores.append(score)
            
            if not mmr_scores:
                break
                
            next_idx = remaining[np.argmax(mmr_scores)]
            selected.append(next_idx)
            remaining.remove(next_idx)

        # 返回原始 chunks（通过 top_indices 映射回去）
        return [self.chunks[top_indices[i]] for i in selected]

# ----------------------------------------------------
# Summarizer: LLM summarization + LLM relevance scoring
# ----------------------------------------------------
class Summarizer:
    def __init__(self, summary_model: str = "meta-llama/Llama-3.1-8B-Instruct:novita", score_model: str = "meta-llama/Llama-3.1-8B-Instruct:novita"):
        # 使用 Hugging Face 路由器 API
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ.get("HF_TOKEN", ""),
        )
        self.score_model = score_model      # Llama 3.1 8B for scoring

    def summarize_and_score(self, chunk: Chunk, question: str) -> Evidence:
        """
        Use unified prompt to get both summary and score in one response.
        Uses Llama 3.1 8B for combined summary and scoring.
        """
        prompt = f"""Summarize the text below to help answer a question. Do not directly answer the question, instead summarize to give evidence to help answer the question. Reply 'Not applicable' if
text is irrelevant. Use concise summary length. At the end of your response, provide a score from 1-10 on a newline indicating relevance to question. Do not explain your score.

Excerpt from citation:
{chunk.text}

Question: {question}

Relevant Information Summary:
"""
        response = self.client.chat.completions.create(
            model=self.score_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = response.choices[0].message.content.strip()
        
        # Parse summary and score from response
        if "\n" in content:
            *summary_lines, last_line = content.split("\n")
            summary = "\n".join(summary_lines).strip()
            # Extract score from last line
            match = re.search(r"(\d+(\.\d+)?)", last_line)
            score = float(match.group(1)) if match else None
        else:
            summary = content
            score = None

        return Evidence(chunk_id=chunk.chunk_id, summary=summary, score=score)


# ----------------------------------------------------
# GatherTool: orchestrates retrieval → summarize → score
# ----------------------------------------------------
class GatherTool:
    def __init__(self, retriever: Retriever, summarizer: Summarizer, embedder: Optional[SentenceTransformer] = None):
        self.retriever = retriever
        self.summarizer = summarizer
        self.embedder = embedder
        if self.embedder is None:
            self.embedder = SentenceTransformer("BAAI/bge-base-en")

    def gather(self, question: str, query_emb: Optional[np.ndarray] = None) -> List[Evidence]:
        """
        1️⃣ Retrieve diverse chunks (MMR)
        2️⃣ Summarize each chunk with LLM (1–10 scoring)
        3️⃣ Sort by score
        Args:
            question: The question to answer
            query_emb: Optional pre-computed query embedding. If None, will generate from question.
        """
        # Generate query embedding if not provided
        if query_emb is None:
            query_emb = self.embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
        
        # Step 1 — MMR retrieval
        chunks: List[Chunk] = self.retriever.mmr_search(query_emb, k=EVIDENCE_TOPK_CONTEXT)
 
        # Step 2 — Concurrent summarization + scoring
        evidences: List[Evidence] = self._batch_summarize(chunks, question)

        # Step 3 — Sort by score (filter out None scores first)
        evidences = [e for e in evidences if e.score is not None]
        evidences.sort(key=lambda e: e.score, reverse=True)
        return evidences

    def _batch_summarize(self, chunks: List[Chunk], question: str) -> List[Evidence]:
        """Run summarization + scoring concurrently."""
        from contextlib import redirect_stderr
        from io import StringIO
        
        evidences = []
        # 临时重定向 stderr 以抑制 tokenizers 警告
        stderr_buffer = StringIO()
        with redirect_stderr(stderr_buffer):
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(self.summarizer.summarize_and_score, c, question) for c in chunks]
                for f in futures:
                    evidences.append(f.result())
        return evidences


if __name__ == "__main__":
    from paperqa.tools.tool_search import SearchTool
    
    async def test_gather_evidence():
        try:
            litqa_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "litqa-v0.jsonl")
            with open(litqa_path, "r") as f:
                samples = [json.loads(line) for line in f if line.strip()]
            # 过滤掉没有 question 字段的行（如 canary 行）
            samples = [s for s in samples if "question" in s]
            if not samples:
                print("❌ No valid questions found in litqa-v0.jsonl")
                return
            
            # 随机选择一个问题
            random.seed(123)  # 使用当前时间作为随机种子
            sample = random.choice(samples)
            question = sample["question"]
            
            print(f"\n🧪 Testing LitQA Question:\n{question}\n")
            if "ideal" in sample:
                print(f"📝 Ideal Answer: {sample['ideal']}\n")
            
            search_tool = SearchTool(None, None, None, None, None, None, None)
            hits = await search_tool.smart_search(question, min_hits=3, max_rounds=2)
            print(f"📚 Found {len(hits)} papers")
            
            chunks = await search_tool.ingest(hits)
            print(f"📄 Generated {len(chunks)} chunks\n")
            
            if not chunks:
                print("⚠️ No chunks available")
                return
            
            retriever = Retriever(chunks)
            summarizer = Summarizer(summary_model="meta-llama/Llama-3.1-8B-Instruct:novita", score_model="meta-llama/Llama-3.1-8B-Instruct:novita")
            gather_tool = GatherTool(retriever, summarizer, embedder=search_tool.embedder)
            
            print("🔍 Gathering evidence...\n")
            evidences = gather_tool.gather(question)
            
            print(f"✅ Collected {len(evidences)} evidences:\n")
            # 创建 chunk_id 到 chunk 的映射，方便查看原文
            chunk_dict = {chunk.chunk_id: chunk for chunk in chunks}
            
            for i, ev in enumerate(evidences[:10], 1):
                print(f"[{i}] Score: {ev.score}/10")
                print(f"    Chunk ID: {ev.chunk_id[:60]}...")
                
                # 显示原始 chunk 文本
                if ev.chunk_id in chunk_dict:
                    original_text = chunk_dict[ev.chunk_id].text[:-1].replace('\n', ' ')
                    print(f"    Original Text: {original_text}...")
            
                print(f"    Summary: {ev.summary[:-1]}...")
                
                print()
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(test_gather_evidence())
