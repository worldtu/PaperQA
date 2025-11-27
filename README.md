# PaperQA

A question-answering system for scientific papers based on the PaperQA research framework. The system retrieves relevant papers, extracts evidence, and generates answers with citations.

## Project Structure

```
paperqa/
├── agent/
│   ├── agent.py          # PaperQAAgent - Main orchestrator
│   ├── policy.py          # Agent policy definitions
│   └── registry.py        # Tool registry
├── tools/
│   ├── tool_ask.py       # Ask Tool - Background knowledge generation
│   ├── tool_search.py     # Search Tool - Paper search and ingestion
│   ├── tool_gather.py     # Gather Tool - Evidence retrieval (MMR + BM25)
│   └── tool_answer.py     # Answer Tool - Final answer generation
├── app.py                 # Streamlit UI application
├── schemas.py             # Pydantic data models (Chunk, Answer, SearchHit, etc.)
└── settings.py            # Configuration parameters
```

## Architecture

The system consists of a **PaperQAAgent** that orchestrates four main tools:

1. **Ask Tool** - Generates background knowledge
2. **Search Tool** - Finds and processes relevant papers
3. **Gather Tool** - Retrieves relevant evidence chunks
4. **Answer Tool** - Generates final answers

## Workflow

```
Question
  ↓
[Ask] Generate background knowledge (once)
  ↓
[Search] Find relevant papers (arXiv)
  ↓
[Ingest] Download PDFs → Extract text → Chunk → Generate embeddings
  ↓
[Gather] Hybrid retrieval (Semantic + BM25) → MMR selection → Optional cleaning
  ↓
[Answer] Generate answer with citations
  ↓
Check if sufficient → Iterate if needed
```

## Tools

### 1. Ask Tool (`tool_ask.py`)
**Function**: Generate parametric background knowledge
- Uses LLM to generate 40-50 word background knowledge
- Provides context to help detect contradictions
- Called once per question

**Key Method**:
- `generate_background(question: str) -> str`

### 2. Search Tool (`tool_search.py`)
**Functions**:
- **Search**: Find relevant papers from arXiv
  - Extracts keywords from question
  - Searches arXiv API (max 20 results)
  - Reranks by semantic similarity (top 3)
- **Ingest**: Process papers into chunks
  - Downloads PDFs
  - Extracts text using PyMuPDF
  - Splits into chunks (size=1000, overlap=0.2)
  - Generates embeddings (BAAI/bge-base-en)
  - Caches chunks and embeddings

**Key Methods**:
- `search(question: str) -> List[SearchHit]`
- `ingest(hits: List[SearchHit]) -> List[Chunk]`

### 3. Gather Tool (`tool_gather.py`)
**Functions**:
- **Hybrid Retrieval**: Combines semantic search (70%) and BM25 (30%)
- **MMR Selection**: Maximal Marginal Relevance algorithm
  - Balances relevance and diversity (λ=0.8)
  - Returns top-k diverse chunks (k=5)
- **Optional Cleaning**: LLM-based text cleaning (parallel processing)

**Key Methods**:
- `gather(question: str) -> List[Evidence]`
- `mmr_search(query_emb, query_text, k) -> List[Chunk]`

### 4. Answer Tool (`tool_answer.py`)
**Function**: Generate final answer with citations
- Combines context chunks and background knowledge
- Uses LLM to generate structured JSON response
- Returns answer, citations, confidence score, and need_more flag

**Key Method**:
- `answer(question, context_chunks, background) -> Answer`

## Agent Usage

### Initialization

```python
from agent.agent import PaperQAAgent
from tools.tool_search import SearchTool
from tools.tool_answer import AnswerLLMTool
from tools.tool_ask import AskTool

# Initialize tools
search_tool = SearchTool(...)
ask_tool = AskTool(model="llama3.1:8b")
answer_tool = AnswerLLMTool(model="llama3.1:8b")

# Initialize agent
agent = PaperQAAgent(
    search_tool=search_tool,
    answer_tool=answer_tool,
    ask_tool=ask_tool,
    max_iterations=2,
    use_chunk_cleaning=False
)
```

### Running the Agent

```python
import asyncio

result = await agent.run(question="What is the attention mechanism?")

# Returns:
# {
#     'question': str,
#     'answer': str,
#     'confidence': float,
#     'citations': List[str],
#     'iterations': int,
#     'first_chunk': str,
#     'background': str
# }
```

## Agent Workflow Details

1. **Ask** (once): Generate background knowledge using `ask_tool.generate_background()`
2. **Iterative Loop** (max_iterations times):
   - **Search**: Find papers using `search_tool.search()`
   - **Ingest**: Process PDFs using `search_tool.ingest()`
   - **Gather**: Retrieve evidence using `gather_tool.gather()`
   - **Answer**: Generate answer using `answer_tool.answer()`
   - **Check**: If answer is sufficient (confidence ≥ 0.7, has citations, need_more=False), stop; otherwise continue

## Configuration

Key settings in `settings.py`:
- `EVIDENCE_TOPK_CONTEXT = 5` - Number of evidence chunks to retrieve
- `MMR_LAMBDA = 0.8` - MMR relevance weight (higher = more relevance)
- `CHUNK_SIZE = 1000` - Text chunk size
- `CHUNK_OVERLAP = 0.2` - Chunk overlap ratio
- `MAX_ITERATIONS = 2` - Maximum search iterations

## Requirements

- Python 3.8+
- Ollama (for LLM inference)
- Required packages: `arxiv`, `sentence-transformers`, `rank-bm25`, `PyMuPDF`, `openai`, `streamlit`

## UI

Run the Streamlit UI:
```bash
streamlit run app.py
```

The UI provides:
- Question input
- Real-time workflow visualization
- Final answer with citations
- Context information (background, chunks)
