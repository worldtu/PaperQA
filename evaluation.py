"""
Unified evaluation entry for PaperQA.

Two modes:
  1) ask_only: Evaluate Ask LLM (<=50 words background) on LAB-Bench/LitQA2 questions
  2) pipeline: Run end-to-end pipeline (Search→Gather→Ask→Answer) on a few questions

Usage examples (PowerShell):
  py -m pip install -r requirements.txt
  $env:GEMINI_API_KEY="your-gemini-key"

  # Evaluate Ask module on LitQA2 (first 20 samples)
  py evaluation.py --mode ask_only --limit 20

  # Evaluate full pipeline on 5 samples (slower)
  py evaluation.py --mode pipeline --limit 5
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, Any

from datasets import load_dataset

from paperqa.tools.tool_ask import run_tool as run_ask
from paperqa.pipeline import Pipeline


def eval_ask_only(limit: int = 20) -> None:
    ds = load_dataset("futurehouse/lab-bench", "LitQA2")
    data = ds["train"]
    n = min(limit, len(data))

    total_t = 0.0
    print(f"Ask-only evaluation on LitQA2 (n={n})")
    print("=" * 60)

    for i in range(n):
        q = data[i]["question"]
        t0 = time.time()
        bk = run_ask({"id": f"q{i+1}", "text": q})
        dt = time.time() - t0
        total_t += dt
        print(f"[{i+1}] {q[:120]}...")
        print(f"  background(<=50): {bk.get('background_text','')}")
        print(f"  confidence: {bk.get('confidence_score')}, time: {dt:.2f}s\n")

    if n:
        print("Summary:")
        print(f"  avg latency: {total_t/n:.2f}s")


def eval_pipeline(limit: int = 5) -> None:
    ds = load_dataset("futurehouse/lab-bench", "LitQA2")
    data = ds["train"]
    n = min(limit, len(data))

    pipe = Pipeline()
    total_t = 0.0
    print(f"Pipeline evaluation on LitQA2 (n={n})")
    print("=" * 60)

    for i in range(n):
        q = data[i]["question"]
        out = pipe.run(q)
        total_t += out.get("latency_s", 0.0)

        bk = out.get("background", {})
        ev = out.get("evidence", {})
        ev_list = ev.get("evidence") if isinstance(ev, dict) else None
        ans = out.get("answer", {})

        print(f"[{i+1}] {q[:120]}...")
        print(f"  background(<=50): {bk.get('background_text','')}")
        if ev_list:
            top = ev_list[:3]
            for j, e in enumerate(top, 1):
                print(f"  ev{j} (score={e.get('score','NA')}): {e.get('summary','')[:160]}...")
        else:
            print("  ev: (none)")
        if isinstance(ans, dict):
            print(f"  answer: {(ans.get('answer_text') or str(ans))[:200]}...")
        else:
            print(f"  answer: {str(ans)[:200]}...")
        print(f"  latency: {out.get('latency_s')}s\n")

    if n:
        print("Summary:")
        print(f"  avg latency: {total_t/n:.2f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ask_only", "pipeline"], required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    if args.mode == "ask_only":
        eval_ask_only(args.limit)
    else:
        eval_pipeline(args.limit)


if __name__ == "__main__":
    main()

import json
import sys
from settings import Settings
from pipeline import PaperQAPipeline  # the object created by person A

def print_evaluation_results(results, metrics, format_type: str = "text") -> None:
    """Print evaluation results in specified format."""
    if format_type == "json":
        output = {
            "results": [result.dict() for result in results],
            "metrics": metrics.dict()
        }
        print(json.dumps(output, indent=2))
    else:
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Total Questions: {metrics.total_questions}")
        print(f"Correct:         {metrics.correct}")
        print(f"Incorrect:       {metrics.incorrect}")
        print(f"Unsure:          {metrics.unsure}")
        print(f"\nAccuracy:        {metrics.accuracy:.1%} (Correct/All)")
        print(f"Precision:       {metrics.precision:.1%} (Correct/Sure)")
        print("="*60)


def run_evaluation(pipeline: PaperQAPipeline, dataset_size: int, 
                   debug: bool, format_type: str = "text") -> None:
    """Run evaluation on the dataset."""
    try:
        from datasets import load_dataset
        
        print(f"Loading evaluation dataset...")
        ds = load_dataset(Settings.EVALUATION_DATASET, Settings.EVALUATION_SPLIT)
        ds.set_format(type='pandas')
        
        # Limit dataset size
        questions = []
        correct_answers = []
        
        for i in range(min(dataset_size, len(ds['train']))):
            question_data = ds['train'][i]
            correct_answer = question_data['ideal'][0]
            distractors = question_data['distractors'][0].tolist()
            question_text = question_data['question'][0]
            
            # Create multiple choice question
            candidates = distractors + [correct_answer]
            full_question = f"{question_text} Select multiple-choice options as answers (two at most): {candidates}"
            
            questions.append(full_question)
            correct_answers.append(correct_answer)
        
        print(f"Evaluating {len(questions)} questions...")
        
        # Run evaluation
        results = pipeline.evaluate_questions(questions, correct_answers, debug)
        metrics = pipeline.compute_metrics(results)
        
        print_evaluation_results(results, metrics, format_type)
        
    except ImportError:
        print("Error: datasets library not installed. Install with: pip install datasets")
        sys.exit(1)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)

        sys.exit(1)
