# PaperQA

A multi-agent question-answering system for scientific papers, implementing the framework from [PaperQA: Improving the Answer Quality of LLMs via Multi-Agent Collaboration](https://arxiv.org/abs/2312.07559).

## Overview

PaperQA uses a 4-person collaborative framework where specialized agents work together to answer questions about scientific literature:

- **Person A (Search)**: Searches academic database arXiv
- **Person B (Gather)**: Retrieves and summarizes relevant evidence using MMR
- **Person C (Ask)**: Provides background context with domain knowledge
- **Person D (Answer)**: Synthesizes final answers with citations

## Installation

### Prerequisites

- Python 3.12+
- Java 11+ (for Pyserini)
- PyTorch (install separately for your platform)

### Setup

```bash
# Install PyTorch first (choose appropriate version)
pip install torch==2.4.0  # macOS

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run the main pipeline
python main.py

# Evaluate Answer LLM on sample test data
python tests/test_answer_llm.py

# Evaluate with limited questions
python tests/test_answer_llm.py --max-questions 10 --debug
```

## Project Structure

```
PaperQA/
├── agent/
│   ├── agent.py          # Main agent orchestration
│   ├── policy.py          # Stop rules & action selection
│   └── registry.py        # Tool registry & dependency injection
├── tools/
│   ├── tool_search.py     # Person A: Search Tool
│   ├── tool_gather.py     # Person B: Gather Evidence
│   ├── tool_ask.py        # Person C: Ask LLM (background)
│   └── tool_answer.py     # Person D: Answer LLM (final answer)
├── tests/
│   ├── test_answer_llm.py           # Answer LLM evaluation script
│   └── answer_llm_evaluation_results.json  # Evaluation results
├── data/
│   ├── PaperQA Golden Test Data.csv  # Test dataset
│   └── cache/                        # Cached API responses & PDFs
├── schemas.py            # Pydantic schemas (Evidence, Answer, etc.)
├── settings.py            # Configuration (models, paths, parameters)
├── evaluation.py         # Evaluation framework
├── main.py               # Entry point
└── requirements.txt      # Dependencies
```

## Configuration

Edit `settings.py` to configure:

- **Models**: LLM and embedding model names
- **Paths**: Data directories, cache locations
- **RAG Parameters**: Chunk size, overlap, MMR lambda
- **Generation Parameters**: Temperature, top-p, top-k, max tokens

```python
# Example: Change model
Settings.MODEL_NAME = "your-model-name"
```

## Evaluation

### Answer LLM

The Answer LLM can be evaluated on golden test data:

```bash
# Answer evaluation
python tests/test_answer_llm.py

# Options:
#   --csv PATH              # Custom CSV test file
#   --max-questions N       # Limit number of questions
#   --debug                 # Enable debug logging
```

**Output:**
- Console: Real-time progress bar with statistics
- Log file: `data/answer_llm_evaluation_{timestamp}.log`
- Results JSON: `tests/answer_llm_evaluation_results.json`

**Metrics:** Accuracy, Precision, Recall, F1 Score (exact/partial match)

## Key Features

- 🔍 **Multi-source Search**: arXiv, PubMed, Google Scholar integration
- 📊 **MMR-based Retrieval**: Maximum Marginal Relevance for diverse evidence
- 🤖 **Multi-agent Collaboration**: Specialized agents for each task
- 📝 **Citation Support**: Answers include source citations
- ✅ **Comprehensive Evaluation**: Test suite with golden data
- 📈 **Progress Tracking**: Real-time evaluation with tqdm progress bars

## Architecture

The system follows a pipeline architecture:

1. **Question Input** → Agent receives question
2. **Search Phase** → Person A searches academic databases
3. **Gather Phase** → Person B retrieves and summarizes evidence
4. **Background Phase** → Person C provides domain context
5. **Answer Phase** → Person D synthesizes final answer with citations

Each phase uses specialized LLM prompts and retrieval techniques optimized for its task.

## Requirements

See `requirements.txt` for full list. Key dependencies:

- `transformers` - Hugging Face models
- `sentence-transformers` - Embeddings
- `faiss-cpu` - Vector similarity search
- `pyserini` - Keyword search (requires Java 11+)
- `tqdm` - Progress bars

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{paperqa2023,
  title={PaperQA: Improving the Answer Quality of LLMs via Multi-Agent Collaboration},
  author={...},
  journal={arXiv preprint arXiv:2312.07559},
  year={2023}
}
```
