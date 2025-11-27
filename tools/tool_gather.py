"""
PaperQA — Gather Evidence Tool
Implements: MMR retrieval + optional chunk cleaning
"""
import os
import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*tokenizers.*")

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from schemas import Chunk, Evidence
from settings import (
    EVIDENCE_TOPK_CONTEXT,
    MMR_LAMBDA,
)

# Try to import BM25, install with: pip install rank-bm25
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
# ----------------------------------------------------
# Retriever: use MMR to select diverse & relevant chunks
# ----------------------------------------------------
class Retriever:
    def __init__(self, chunks: List[Chunk], lambda_param: float = MMR_LAMBDA, use_bm25: bool = True, bm25_weight: float = 0.3):
        """
        Initialize Retriever
        
        Args:
            chunks: List of chunks to search from
            lambda_param: MMR lambda parameter (0-1), higher = more relevance, lower = more diversity
            use_bm25: Whether to use BM25 for hybrid search (default True)
            bm25_weight: Weight for BM25 score in hybrid relevance (0-1), semantic weight = 1 - bm25_weight
        """
        self.chunks = chunks
        self.lambda_param = lambda_param
        self.use_bm25 = use_bm25 and BM25_AVAILABLE
        self.bm25_weight = bm25_weight
        
        # Initialize BM25 if available
        if self.use_bm25:
            # Tokenize chunks for BM25
            self.tokenized_chunks = [chunk.text.lower().split() for chunk in self.chunks]
            self.bm25 = BM25Okapi(self.tokenized_chunks)
    
    def mmr_search(self, query_emb: np.ndarray, query_text: str = "", k: int = EVIDENCE_TOPK_CONTEXT) -> List[Chunk]:
        """
        Return top-k diverse and relevant chunks using Maximal Marginal Relevance (MMR).
        Optionally combines semantic search with BM25 for hybrid retrieval.
        
        Args:
            query_emb: Query embedding vector
            query_text: Original query text (for BM25, optional)
            k: Number of chunks to return
        """
        chunks_without_embedding = [c for c in self.chunks if c.embedding is None]
        if chunks_without_embedding:
            raise ValueError(f"{len(chunks_without_embedding)} chunks missing embeddings. "
                           "Embeddings should be generated in SearchTool.ingest().")
        
        if len(query_emb.shape) > 1:
            query_emb = query_emb.flatten()
        
        # Compute semantic similarity (cosine similarity)
        docs = np.stack([np.array(c.embedding) for c in self.chunks])
        semantic_scores = np.dot(docs, query_emb) / (
            np.linalg.norm(docs, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )
        
        # Compute hybrid relevance score (semantic + BM25)
        if self.use_bm25 and query_text:
            # Get BM25 scores
            query_tokens = query_text.lower().split()
            bm25_scores = self.bm25.get_scores(query_tokens)
            
            # Normalize both scores to [0, 1]
            semantic_norm = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-8)
            bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
            
            # Hybrid relevance: combine semantic and BM25
            relevance_scores = (1 - self.bm25_weight) * semantic_norm + self.bm25_weight * bm25_norm
        else:
            # Use only semantic scores
            relevance_scores = semantic_scores
        
        # Compute chunk-to-chunk similarity matrix (for diversity)
        sim_matrix = np.dot(docs, docs.T)
        
        # MMR selection
        selected, remaining = [], list(range(len(self.chunks)))
        
        while len(selected) < k and remaining:
            mmr_scores = []
            for i in remaining:
                # Relevance to query (hybrid: semantic + BM25)
                relevance = relevance_scores[i]
                # Maximum redundancy with already selected chunks
                redundancy = max([sim_matrix[i, j] for j in selected], default=0)
                # MMR score: balance relevance and diversity
                score = self.lambda_param * relevance - (1 - self.lambda_param) * redundancy
                mmr_scores.append(score)
            
            if not mmr_scores:
                break
                
            next_idx = remaining[np.argmax(mmr_scores)]
            selected.append(next_idx)
            remaining.remove(next_idx)
        
        # Return selected chunks
        return [self.chunks[i] for i in selected]

# ----------------------------------------------------
# GatherTool: orchestrates retrieval → clean chunks
# ----------------------------------------------------
class GatherTool:
    def __init__(self, retriever: Retriever, embedder: Optional[SentenceTransformer] = None, 
                 clean_model: str = "llama3.1:8b", use_cleaning: bool = True):
        """
        Initialize GatherTool
        
        Args:
            retriever: MMR retriever
            embedder: Embedding model
            clean_model: LLM model for cleaning chunk text
            use_cleaning: Whether to use LLM for cleaning chunk text (default True)
        """
        self.retriever = retriever
        self.embedder = embedder or SentenceTransformer("BAAI/bge-base-en")
        self.use_cleaning = use_cleaning
        
        if self.use_cleaning:
            self.client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            )
            self.clean_model = clean_model
        else:
            self.client = None
            self.clean_model = None
    
    def clean_chunk(self, chunk: Chunk) -> str:
        """
        Clean chunk text using LLM to remove formatting issues
        
        Args:
            chunk: Original chunk
            
        Returns:
            str: Cleaned text
        """
        prompt = f"""Fix the formatting of this text excerpt from a scientific paper. Remove line breaks and spacing issues, fix broken sentences, but keep ALL information exactly as it is.

Text:
{chunk.text}

IMPORTANT: Return ONLY the cleaned text content. Do NOT add any explanations, prefixes like "Here is..." or other commentary. Just return the cleaned paragraph directly.

Output:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.clean_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000,
            )
            cleaned = response.choices[0].message.content.strip()
            return cleaned if cleaned else chunk.text
        except Exception:
            return chunk.text

    def gather(self, question: str, query_emb: Optional[np.ndarray] = None) -> List[Evidence]:
        """
        Retrieve top-K relevant chunks using MMR, optionally clean chunk text
        
        Args:
            question: The question to answer
            query_emb: Optional pre-computed query embedding. If None, will generate from question.
        
        Returns:
            List[Evidence]: List of evidence with chunk_id and similarity score
        """
        # Generate query embedding if not provided
        if query_emb is None:
            query_emb = self.embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
        
        # Step 1: MMR retrieval (with BM25 hybrid search if enabled)
        chunks: List[Chunk] = self.retriever.mmr_search(query_emb, query_text=question, k=EVIDENCE_TOPK_CONTEXT)
        
        # Step 2: Clean chunk text (if enabled, parallel processing)
        if self.use_cleaning:
            def clean_single_chunk(chunk_idx_chunk):
                idx, chunk = chunk_idx_chunk
                cleaned_text = self.clean_chunk(chunk)
                return idx, cleaned_text
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(clean_single_chunk, (i, chunk)): i 
                          for i, chunk in enumerate(chunks)}
                for future in as_completed(futures):
                    idx, cleaned_text = future.result()
                    chunks[idx].text = cleaned_text
 
        # Step 3: Convert to Evidence objects (compute similarity scores)
        evidences = []
        for chunk in chunks:
            chunk_emb = np.array(chunk.embedding)
            similarity = np.dot(chunk_emb, query_emb) / (
                np.linalg.norm(chunk_emb) * np.linalg.norm(query_emb) + 1e-8
            )
            evidences.append(Evidence(
                chunk_id=chunk.chunk_id,
                summary="",
                score=float(similarity * 10)
            ))
        
        # Sort by relevance score (descending)
        evidences.sort(key=lambda x: x.score, reverse=True)
        
        return evidences

