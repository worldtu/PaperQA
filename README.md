# PaperQA
Implementation on PaperQA based on article: arxiv.org/abs/2312.07559

## Repo Structure

```
paperqa/
├─ agent/
│  ├─ agent.py                 # PaperQA agent (policy + control loop)
│  ├─ policy.py                # Stop rules & action-selection policy
│  └─ registry.py              # Tool registry & dependency injection
├─ tools/
│  ├─ tool_search.py           # Person A — Search Tool (Scholar+arXiv+PubMed)
│  ├─ tool_gather.py           # Person B — Gather Evidence (MMR + Summary LLM)
│  ├─ tool_ask.py              # Person C — Ask LLM (a priori background)
│  └─ tool_answer.py           # Person D — Answer LLM (reduce + citations)
├─ data/
│  ├─ cache/                   # HTTP/cache for APIs and parsed PDFs
│  ├─ store.sqlite             # Metadata & schema objects (optional)
│  └─ faiss.index              # Vector index (FAISS)
├─ schemas.py                  # Shared JSON/Pydantic schemas (interfaces)
├─ settings.py                 # Model names, batch sizes, MMR lambda, thresholds
├─ pipeline.py                 # Wiring: queues, async pools, and shared state
├─ requirements.txt
└─ main.py                     # CLI entry: `python -m paperqa.main --q "..."` 
```