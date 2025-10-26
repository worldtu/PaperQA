#!/usr/bin/env python3
"""
Complete example showing how to run AnswerTool with real LLM integration.
This demonstrates the full PaperQA answer generation process.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.tool_answer import AnswerTool, Evidence, Background
from tools.tool_llm import (
    create_gpt2,
    create_deepseek_r1_distill_qwen_15b,
    create_mistral_7b,
    MockLLM,
)

def create_sample_evidence():
    """Create realistic sample evidence for testing."""
    return [
        Evidence(
            chunk_id="chunk_001",
            summary="Transformer models use self-attention mechanisms to process sequences of tokens, allowing them to capture long-range dependencies in text.",
            score=9.1,
            citation="(Vaswani2017)"
        ),
        Evidence(
            chunk_id="chunk_002", 
            summary="The encoder-decoder architecture with multi-head attention has become the standard for neural machine translation and language understanding tasks.",
            score=8.7,
            citation="(Devlin2018)"
        ),
        Evidence(
            chunk_id="chunk_003",
            summary="Pre-training on large corpora followed by task-specific fine-tuning enables transfer learning across diverse NLP applications.",
            score=8.9,
            citation="(Radford2019)"
        ),
        Evidence(
            chunk_id="chunk_004",
            summary="Positional encoding allows transformers to understand the order of tokens in sequences, which is crucial for language understanding.",
            score=8.3,
            citation="(Shaw2018)"
        ),
        Evidence(
            chunk_id="chunk_005",
            summary="Layer normalization and residual connections help stabilize training of deep transformer networks with many layers.",
            score=7.8,
            citation="(Ba2016)"
        ),
        Evidence(
            chunk_id="chunk_006",
            summary="Recent work has shown that scaling up transformer models with more parameters and training data leads to emergent capabilities.",
            score=9.0,
            citation="(Brown2020)"
        )
    ]

def run_with_huggingface_small():
    """Run AnswerTool with small Hugging Face model (fast, low memory)."""
    print("=== Running with Hugging Face (Small Model) ===")
    print("Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    
    try:
        # Create small Hugging Face models
        ask_llm = create_deepseek_r1_distill_qwen_15b()
        answer_llm = create_deepseek_r1_distill_qwen_15b()
        
        # Create AnswerTool
        tool = AnswerTool(
            ask_llm=ask_llm,
            answer_llm=answer_llm,
            answer_length="medium"
        )
        
        # Sample question and evidence
        question = "What are the key innovations in transformer architecture?"
        evidences = create_sample_evidence()
        
        # Generate answer
        answer = tool.generate(question, evidences)
        
        print(f"Question: {question}")
        print(f"\nAnswer: {answer.text}")
        print(f"\nCitations: {', '.join(answer.citations)}")
        print(f"Confidence: {answer.confidence:.2f}")
        
    except Exception as e:
        print(f"Hugging Face small model error: {e}")
        print("Falling back to mock LLM...")
        MockLLM()

def run_with_huggingface_medium():
    """Run AnswerTool with medium Hugging Face model (balanced quality/speed)."""
    print("=== Running with Hugging Face (Medium Model) ===")
    print("Model: DialoGPT-medium (balanced quality and speed)")
    
    try:
        # Create medium Hugging Face models
        ask_llm = create_mistral_7b()
        answer_llm = create_mistral_7b()
        
        # Create AnswerTool
        tool = AnswerTool(
            ask_llm=ask_llm,
            answer_llm=answer_llm,
            answer_length="medium"
        )
        
        # Sample question and evidence
        question = "How has transformer architecture evolved since its introduction?"
        evidences = create_sample_evidence()
        
        # Generate answer
        answer = tool.generate(question, evidences)
        
        print(f"Question: {question}")
        print(f"\nAnswer: {answer.text}")
        print(f"\nCitations: {', '.join(answer.citations)}")
        print(f"Confidence: {answer.confidence:.2f}")
        
    except Exception as e:
        print(f"Hugging Face medium model error: {e}")
        print("Falling back to mock LLM...")
        MockLLM()


def demonstrate_paper_workflow():
    """Demonstrate the complete PaperQA workflow as described in the paper."""
    print("=== PaperQA Workflow Demonstration ===")
    print("Following the exact specification from the paper...")
    
    # Step 1: Create AnswerTool with proper configuration
    ask_llm = create_gpt2()
    answer_llm = create_gpt2()
    
    tool = AnswerTool(
        ask_llm=ask_llm,
        answer_llm=answer_llm,
        answer_length="medium",
        min_evidence=5,
        top_k=8,
        min_avg_score=6.5
    )
    
    # Step 2: Sample question (from the paper's example)
    question = "What are the main components of transformer architecture?"
    
    # Step 3: Evidence from context library (gathered by GatherTool)
    evidences = create_sample_evidence()
    
    # Step 4: Background information (from AskTool)
    background = Background(
        question=question,
        background_text="Transformers are a type of neural network architecture that has become the foundation for modern NLP models."
    )
    
    print(f"Question: {question}")
    print(f"Number of evidence chunks: {len(evidences)}")
    print(f"Average evidence score: {sum(e.score for e in evidences) / len(evidences):.1f}")
    print()
    
    # Step 5: Generate answer following paper specification
    answer = tool.generate(question, evidences, background)
    
    print("Generated Answer:")
    print("-" * 50)
    print(answer.text)
    print("-" * 50)
    print(f"Citations: {', '.join(answer.citations)}")
    print(f"Confidence: {answer.confidence:.2f}")
    print()

def main():
    """Main function to run all examples."""
    print("PaperQA AnswerTool Demonstration")
    print("=" * 70)
    print()
    
    # Run different configurations
    print("Testing with models...")
    print()
    
    # Traditional models
    # run_with_huggingface_small()
    # run_with_huggingface_medium()

    # PaperQA workflow demonstration
    demonstrate_paper_workflow()
    

if __name__ == "__main__":
    main()
