# paperqa/tools/tool_search.py
import os
import arxiv
import requests
import fitz
import asyncio
from typing import List
from schemas import SearchHit, Chunk
from settings import CHUNK_SIZE, CHUNK_OVERLAP
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import json

# make sure cache folder exists
os.makedirs("data/cache", exist_ok=True)


class SearchTool:
    def __init__(self, scholar_client, arxiv_client, pubmed_client, parser, embedder, indexer, cache):
        self.scholar = scholar_client
        self.arxiv = arxiv_client
        self.pubmed = pubmed_client
        self.parser = parser
        self.embedder = SentenceTransformer("BAAI/bge-base-en")
        self.cache = cache

    def _extract_keywords(self, question: str) -> str:
        """
        Extract key terms from question for better arXiv search.
        Removes question words and extracts important technical terms.
        Preserves original case (uppercase/lowercase) for proper nouns and acronyms.
        """
        import re
        
        # Remove question words (case-insensitive matching)
        question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'does', 'do', 'is', 'are', 'can', 'could', 'should', 'would', 'will', 'the', 'a', 'an', 'to', 'of', 'in', 'on', 'at', 'for', 'with', 'through', 'enables', 'enable', 'learn', 'learning']
        question_words_set = set(w.lower() for w in question_words)
        
        # Remove quotes and special characters (preserve case)
        text = question.replace('"', '').replace("'", '')
        
        # Split into words
        words = re.findall(r'\b\w+\b', text)
        
        # Filter out question words (case-insensitive) and short words
        keywords = [w for w in words if w.lower() not in question_words_set and len(w) > 3]
        
        # Join with OR for arXiv search (broader matching)
        if keywords:
            return ' OR '.join(keywords[:10])  # Limit to 10 keywords
        else:
            # Fallback: use original question
            return question
    
    async def search(self, question: str, year_hint: str | None = None) -> List[SearchHit]:
        """
        Main search entrypoint.
        Extracts keywords, performs arXiv discovery, and reranks by semantic similarity.
        """
        # Extract keywords for better search
        query = self._extract_keywords(question)
        if year_hint:
            query = f"{query} {year_hint}"
        
        # Discover candidate papers
        hits = await self._fanout_discovery([query])

        # Rerank by semantic similarity (using original question for better semantic matching)
        ranked_hits = self.rank_by_embedding(question, hits, top_k=3)

        return ranked_hits

    
    async def ingest(self, hits: List[SearchHit], use_cache: bool = True) -> List[Chunk]:
        """
        Process PDFs and generate chunks + embeddings
        
        Args:
            hits: List of search results
            use_cache: Whether to use cache (default True)
        """
        # 生成缓存路径
        cache_key = self.get_cache_key(hits)
        chunks_cache_path = f"data/chunks_{cache_key}.pkl"
        
        # Check cache
        if use_cache:
            cached_chunks = self.load_chunks(chunks_cache_path)
            if cached_chunks:
                return cached_chunks
        
        all_chunks: List[Chunk] = []
        for idx, hit in enumerate(hits[:5], 1):  # limit 5 PDFs
            pdf_url = hit.urls.get("pdf")
            if not pdf_url:
                continue

            safe_id = (
                hit.paper_id.replace("http://", "")
                .replace("https://", "")
                .replace("/", "_")
                .replace(":", "_")
            )
            
            pdf_path = self._download_pdf(pdf_url, safe_id)
            if not pdf_path:
                continue

            text = self._pdf_to_text(pdf_path)
            if not isinstance(text, str) or not text.strip():
                continue

            chunks = self._make_chunks(hit.paper_id, text)
            all_chunks.extend(chunks)

        # Generate embeddings and index
        texts = [chunk.text for chunk in all_chunks]
        if texts:
            # Use larger batch_size to speed up embedding generation
            embeddings = self.embedder.encode(
                texts, 
                convert_to_numpy=True, 
                normalize_embeddings=True,
                batch_size=32,  # Increase batch size for faster processing
                show_progress_bar=True
            )
            
            # Attach embeddings to chunks
            for chunk, emb in zip(all_chunks, embeddings):
                chunk.embedding = emb.tolist()
            
            # Save chunks to cache
            if use_cache:
                self.save_chunks(all_chunks, chunks_cache_path)


        return all_chunks

    async def _fanout_discovery(self, queries: List[str]) -> List[SearchHit]:
        temp_hits = []
        for query in queries:
            search = arxiv.Search(
                query=query,
                max_results=20,
                sort_by=arxiv.SortCriterion.Relevance
            )
            for result in search.results():
                temp_hits.append({
                    "paper_id": result.entry_id,
                    "title": result.title,
                    "year": result.published.year if result.published else None,
                    "pdf_url": f"https://arxiv.org/pdf/{result.get_short_id()}.pdf",
                    "abstract": getattr(result, "summary", "")
                })
        # Convert to SearchHit objects (without abstract)
        hits = [
            SearchHit(
                source="arxiv",
                paper_id=h["paper_id"],
                title=h["title"],
                year=h["year"],
                urls={"pdf": h["pdf_url"]},
            )
            for h in temp_hits
        ]
        # Store abstracts in a sidecar dict for reranking
        self._abstract_cache = {h["paper_id"]: h["abstract"] for h in temp_hits}
        return hits

    def rank_by_embedding(self, question: str, hits: List[SearchHit], top_k: int = 5) -> List[SearchHit]:
        """
        Rerank SearchHits by embedding similarity between the user's question
        and each paper's title + abstract (from self._abstract_cache).
        """
        if not hits:
            return []

        # 1️⃣ Combine title + abstract for each paper
        texts = [f"{h.title}. {self._abstract_cache.get(h.paper_id, '')}" for h in hits]

        # 2️⃣ Compute embeddings
        q_vec = self.embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)
        doc_vecs = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        # 3️⃣ Compute cosine similarities
        sims = np.dot(doc_vecs, q_vec.T).squeeze()

        # 4️⃣ Sort papers by similarity
        sorted_pairs = sorted(zip(sims, hits), key=lambda x: x[0], reverse=True)

        # 5️⃣ Keep top_k and attach score for logging
        ranked_hits = []
        for sim, hit in sorted_pairs[:top_k]:
            hit.similarity = float(sim)
            ranked_hits.append(hit)

        return ranked_hits


    def _download_pdf(self, url: str, paper_id: str) -> str:
        """
        Download the PDF for a given paper (only for the top-ranked ones)
        and cache it locally. Returns the local file path if successful,
        else an empty string.
        """
        pdf_dir = os.path.join("data", "cache")
        os.makedirs(pdf_dir, exist_ok=True)
        pdf_path = os.path.join(pdf_dir, f"{paper_id}.pdf")

        # Skip if already cached and non-empty
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000:
            return pdf_path

        try:
            headers = {"User-Agent": "Mozilla/5.0 (PaperQA Bot)"}
            response = requests.get(url, headers=headers, timeout=30)

            # Verify success and non-empty payload
            if response.status_code != 200 or len(response.content) < 1000:
                return ""

            # Save the PDF to cache
            with open(pdf_path, "wb") as f:
                f.write(response.content)

            return pdf_path

        except Exception:
            return ""


    def _pdf_to_text(self, pdf_path: str) -> str:
        try:
            with fitz.open(pdf_path) as doc:
                full_text = "".join(page.get_text("text") for page in doc)
                # 过滤掉 References 部分
                return self._remove_references(full_text)
        except:
            return ""
    
    def _remove_references(self, text: str) -> str:
        """
        Remove References/Bibliography section from paper text
        """
        if not text:
            return text
        
        ref_markers = [
            "\nReferences\n",
            "\nREFERENCES\n",
            "\nBibliography\n",
            "\nBIBLIOGRAPHY\n",
            "\nReferences",
            "\nREFERENCES",
        ]
        
        ref_start = -1
        for marker in ref_markers:
            pos = text.find(marker)
            if pos != -1 and (ref_start == -1 or pos < ref_start):
                ref_start = pos
        
        # If References found, truncate text
        if ref_start != -1:
            return text[:ref_start]
        
        return text

    def _make_chunks(self, paper_id: str, text: str) -> List[Chunk]:
        if not text:
            return []
        chunks = []
        start = 0
        idx = 0
        # 如果 CHUNK_OVERLAP 是比例（0-1），转换为字符数；否则直接使用
        overlap_chars = int(CHUNK_SIZE * CHUNK_OVERLAP) if CHUNK_OVERLAP < 1 else int(CHUNK_OVERLAP)
        
        text_len = len(text)
        step_size = CHUNK_SIZE - overlap_chars
        
        while start < text_len:
            end = min(start + CHUNK_SIZE, text_len)
            chunks.append(
                Chunk(
                    paper_id=paper_id,
                    chunk_id=f"{paper_id}_chunk_{idx}",
                    start=start,
                    end=end,
                    text=text[start:end],
                )
            )
            idx += 1
            
            if end >= text_len:
                break
            
            start = end - overlap_chars
            if start >= end:
                start = end
        
        return chunks

    def save_chunks(self, chunks: List[Chunk], path="data/chunks.pkl"):
        """Save chunks (including embeddings) to file using JSON"""
        json_path = path.replace(".pkl", ".json")
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        chunks_data = [chunk.model_dump() for chunk in chunks]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    
    def load_chunks(self, path="data/chunks.pkl") -> List[Chunk]:
        """Load chunks (including embeddings) from file using JSON"""
        # Try JSON format first
        json_path = path.replace(".pkl", ".json")
        
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    chunks_data = json.load(f)
                # Reconstruct Chunk objects from dictionaries
                chunks = [Chunk.model_validate(chunk_dict) for chunk_dict in chunks_data]
                if chunks and len(chunks) > 0:
                    return chunks
                return []
            except (json.JSONDecodeError, Exception) as e:
                try:
                    os.remove(json_path)
                except:
                    pass
                return []
        
        # Fallback: try old pickle format for backward compatibility
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    chunks = pickle.load(f)
                if chunks and len(chunks) > 0:
                    self.save_chunks(chunks, path)  # Convert to JSON
                    return chunks
                return []
            except (EOFError, pickle.UnpicklingError, Exception) as e:
                try:
                    os.remove(path)
                except:
                    pass
                return []
        
        return []
    
    def get_cache_key(self, hits: List[SearchHit]) -> str:
        """
        生成缓存键：基于 paper_id 列表
        
        提取 arXiv ID 或使用完整 paper_id（避免截断导致冲突）
        """
        cache_parts = []
        for hit in hits[:5]:
            paper_id = hit.paper_id
            
            # 提取 arXiv ID（如果是 arXiv 论文）
            if "arxiv.org/abs/" in paper_id:
                # 提取 arXiv ID (e.g., "2405.21075")
                arxiv_id = paper_id.split("/abs/")[-1].strip()
                cache_parts.append(f"arxiv_{arxiv_id}")
            else:
                # 其他类型的 paper_id，安全处理
                safe_id = paper_id.replace("http://", "").replace("https://", "")
                safe_id = safe_id.replace("/", "_").replace(":", "_")
                # 限制长度但保留足够信息（取前50个字符）
                cache_parts.append(safe_id[:50])
        
        return "_".join(sorted(cache_parts))

