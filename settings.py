# Model names, batch sizes, MMR lambda, thresholds
EMBEDDING_MODEL = "bge-base-en"   # or SciBERT/SPECTER2
SUMMARY_LLM = "meta-llama/Llama-3-8B-Instruct"  # via Ollama or vLLM
ANSWER_LLM = "qwen2.5:14b-instruct"   # reduce step
ASK_LLM = "llama3.1:8b-instruct"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0.2
MMR_LAMBDA = 0.8             # MMR 相关性权重（0-1），越高越重视相关性
MMR_INITIAL_TOPK = 25        # MMR 粗排阶段召回的候选数量（建议设为最终返回数的 5-10 倍）
EVIDENCE_MIN = 5              # stop when ≥ 5 pieces of evidence
EVIDENCE_TOPK_CONTEXT = 5     # 最终返回的证据数量
MAX_AGENT_STEPS = 6           # cap re-search/re-gather loops
TIMEOUT_S = 120