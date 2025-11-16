# PAPERQA/tools/tool_search.py
import os
import arxiv
import requests
import fitz
import asyncio
from typing import List
from ..schemas import SearchHit, Chunk
from ..settings import Settings 
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from nltk.corpus import wordnet as wn

# make sure cache folder exists
os.makedirs("PaperQA/data/cache", exist_ok=True)


class SearchTool:
    def __init__(self, scholar_client, arxiv_client, pubmed_client, parser, embedder, indexer, cache):
        self.scholar = scholar_client
        self.arxiv = arxiv_client
        self.pubmed = pubmed_client
        self.parser = parser
        self.embedder = SentenceTransformer("BAAI/bge-base-en")  # ✅ load embedding model
        self.index_dim = 768  # bge-base-en outputs 768-dim embeddings
        self.indexer = faiss.IndexFlatL2(self.index_dim)  # ✅ simple FAISS index (L2 similarity)
        self.cache = cache

    async def search(self, question: str, year_hint: str | None = None) -> List[SearchHit]:
        """
        Main search entrypoint.
        Expands queries, performs arXiv discovery, and reranks by semantic similarity.
        Output: List[SearchHit] (unchanged)
        """
        # Step 1 — expand query if needed
        variants = self._expand_queries(question, year_hint)

        # Step 2 — discover candidate papers
        hits = await self._fanout_discovery(variants)

        # Step 3 — rerank before returning
        ranked_hits = self.rank_by_embedding(question, hits, top_k=5)

        return ranked_hits
    

    def expand_query_terms(self, question: str, max_terms: int = 5) -> list:
        """
        Expand key terms in the user's query using WordNet synonyms.
        - Extracts meaningful words from the question.
        - For each word, finds up to 2 close synonyms.
        - Returns a list of unique additional terms for reformulation.
        """
        words = [w for w in question.lower().split() if w.isalpha()]
        related = set()

        for w in words:
            for syn in wn.synsets(w):
                for lemma in syn.lemmas()[:2]:  # top 2 synonyms per word
                    name = lemma.name().replace("_", " ")
                    if name != w:
                        related.add(name)

        # return only the top few diverse related terms
        return list(related)[:max_terms]

    async def smart_search(
        self,
        question: str,
        year_hint: str | None = None,
        min_hits: int = 3,
        max_rounds: int = 3
    ):
        """
        🧠 Self-evaluating smart search loop (with WordNet expansion).
        - Expands query terms dynamically using WordNet synonyms.
        - Reformulates queries each round if too few papers are found.
        - Deduplicates results by paper_id.
        - Returns all unique SearchHit objects (silent version).
        """
        all_hits = []
        seen_ids = set()

        # Step 1: Generate initial reformulations using WordNet
        reformulations = self.expand_query_terms(question)
        if not reformulations:
            reformulations = ["review", "overview", "case study"]  # fallback

        # Step 2: Perform multiple rounds of search
        for attempt in range(max_rounds):
            # build a reformulated query
            if attempt < len(reformulations):
                query_variant = f"{question} {reformulations[attempt]}"
            else:
                query_variant = question  # fallback to original

            # perform search
            hits = await self.search(query_variant, year_hint)

            # keep only new hits
            new_hits = [h for h in hits if h.paper_id not in seen_ids]
            for h in new_hits:
                seen_ids.add(h.paper_id)
                all_hits.append(h)

            # stop condition
            if len(all_hits) >= min_hits:
                break

        return all_hits

    
    async def ingest(self, hits: List[SearchHit]) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        for hit in hits[:5]:  # limit 5 PDFs for testing
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

        # ✅ embed chunks and index them
        texts = [chunk.text for chunk in all_chunks]
        if texts:
            embeddings = self.embedder.encode(texts, convert_to_numpy=True)

            # Print a preview vector
            print("\n[EMBEDDING PREVIEW]")
            print(embeddings[0][:10])  # show first 10 numbers of first embedding

            # ✅ add to FAISS index
            self.indexer.add(embeddings)

            self.save_faiss_index()


        return all_chunks

    def _expand_queries(self, q: str, years: str | None) -> List[str]:
        return [q if not years else f"{q} {years}"]

    async def _fanout_discovery(self, queries: List[str]) -> List[SearchHit]:
        temp_hits = []
        for query in queries:
            search = arxiv.Search(
                query=query,
                max_results=10,
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

        print("\n[Semantic Ranking Preview]")
        for i, h in enumerate(ranked_hits):
            print(f"  {i+1}. {h.title[:80]}...  (similarity={h.similarity:.4f})")

        return ranked_hits


    def _download_pdf(self, url: str, paper_id: str) -> str:
        """
        Download the PDF for a given paper (only for the top-ranked ones)
        and cache it locally. Returns the local file path if successful,
        else an empty string.
        """
        pdf_dir = os.path.join("PaperQA", "data", "cache")
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
                return "".join(page.get_text("text") for page in doc)
        except:
            return ""

    def _make_chunks(self, paper_id: str, text: str) -> List[Chunk]:
        if not text:
            return []
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = start + Settings.CHUNK_SIZE
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
            start = end - Settings.CHUNK_OVERLAP
        return chunks

    def save_faiss_index(self, path="PaperQA/data/faiss.index"):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.indexer, path)
        print(f"[FAISS] Index saved to {path}")

    def load_faiss_index(self, path="PaperQA/data/faiss.index"):
        if os.path.exists(path):
            self.indexer = faiss.read_index(path)
            print(f"[FAISS] Index loaded from {path}")
        else:
            print(f"[FAISS] No existing index found at {path}")

    def inspect_faiss(self, k: int = 3):
        """Simple FAISS self-test: loads index, runs a dummy search."""
        try:
            # Load saved index
            self.load_faiss_index()

            print("\n[FAISS] ✅ Index status")
            print(f" - Stored vectors: {self.indexer.ntotal}")
            print(f" - Vector dimension: {self.indexer.d}")

            if self.indexer.ntotal == 0:
                print(" - ⚠️ Index is empty. Run ingest() first.")
                return

            # Create a random query to test search
            import numpy as np
            dummy_query = np.random.random((1, self.indexer.d)).astype("float32")
            distances, ids = self.indexer.search(dummy_query, k)

            print(f" - 🔎 Test query returned {len(ids[0])} nearest neighbors")
            print(f" - Vector IDs: {ids[0]}")
        except Exception as e:
            print(f"[FAISS] ❌ Error inspecting FAISS index: {e}")

if __name__ == "__main__":
    import nltk
    import asyncio
    nltk.download("wordnet", quiet=True)

    async def test_custom():
        tool = SearchTool(None, None, None, None, None, None, None)
        question = "What is the central idea behind the VideoTree framework for LLM video reasoning?"

        # 🧠 WordNet expansions
        related = tool.expand_query_terms(question)
        print(f"\n🧠 WordNet expansions used for SmartSearch:\n   {related if related else '(no expansion found)'}")

        # ✅ Run SmartSearch
        hits = await tool.smart_search(question)
        print(f"\n🧪 Testing Custom Question:\n{question}")

        # 🔍 Print top-ranked semantic results (from _fanout_discovery or ranking)
        print("\n[Semantic Ranking Preview]")
        for i, h in enumerate(hits[:5]):
            title_snippet = h.title[:70] + "..." if len(h.title) > 70 else h.title
            similarity = getattr(h, "similarity", 0.0)
            print(f"  {i+1}. {title_snippet}  (similarity={similarity:.4f})")

        # 📄 Extract and chunk papers
        chunks = await tool.ingest(hits)

        # 🧩 Summary of outputs for next stage
        print(f"\n📚 SearchHit count: {len(hits)}")
        print(f"📄 Chunk count: {len(chunks)}")

    asyncio.run(test_custom())