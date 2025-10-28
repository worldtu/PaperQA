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
