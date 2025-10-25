# paperqa/settings.py
EMBEDDING_MODEL = "bge-base-en"   # or SciBERT/SPECTER2
SUMMARY_LLM = "llama3.1:8b-instruct"  # via Ollama or vLLM
ANSWER_LLM = "qwen2.5:14b-instruct"   # reduce step
ASK_LLM = "llama3.1:8b-instruct"
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 0.2
MMR_LAMBDA = 0.5
EVIDENCE_MIN = 5         # stop when ≥ 5 pieces of evidence
EVIDENCE_TOPK_CONTEXT = 8
MAX_AGENT_STEPS = 6      # cap re-search/re-gather loops
TIMEOUT_S = 120