"""
Test script for evaluating Answer LLM using the golden test data.
Evaluates the model's ability to select correct answer options from multiple choice questions.
"""

import csv
import json
import ast
import sys
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set
from langchain_core.documents import Document
from tqdm import tqdm

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.tool_answer import AnswerLLMTool
from schemas import Evidence, Background, Answer
from settings import Settings


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that uses tqdm.write() to avoid interfering with progress bar."""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stdout)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(log_dir: Path = None, debug: bool = False):
    """Setup logging to both console and file."""
    if log_dir is None:
        log_dir = project_root / "data"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"answer_llm_evaluation_{timestamp}.log"
    
    # Configure logging
    logger = logging.getLogger(__name__)
    log_level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler using tqdm
    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def parse_json_array(value: str) -> List[str]:
    """Parse JSON array string, handling various formats."""
    if not value or value.strip() == '':
        return []
    
    # Replace smart quotes with straight quotes (common in CSV exports from Excel/Google Sheets)
    # Using Unicode escapes to ensure correct character replacement
    value = value.replace('\u201c', '\u0022')  # " → "
    value = value.replace('\u201d', '\u0022')  # " → "
    value = value.replace('\u2018', '\u0027')  # ' → '
    value = value.replace('\u2019', '\u0027')  # ' → '
    
    try:
        # Try parsing as JSON first
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            # Try parsing as Python literal
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        except (ValueError, SyntaxError):
            # Fallback: treat as single string
            return [value.strip()]


def create_mock_evidence_and_documents(
    related_chunks: str,
    reference_paper: str,
    question_no: str
) -> tuple[List[Evidence], List[Document]]:
    """Create mock Evidence and Document objects from test data."""
    evidence_list = []
    documents = []
    
    if not related_chunks or related_chunks.strip() == '':
        # Create a default evidence/document if no chunk provided
        chunk_id = f"test_chunk_{question_no}"
        chunk_text = "No specific chunk provided in test data."
        evidence = Evidence(
            chunk_id=chunk_id,
            summary=chunk_text,
            score=8,
            citation=reference_paper if reference_paper else "Unknown Source"
        )
        doc = Document(
            page_content=chunk_text,
            metadata={
                'chunk_id': chunk_id,
                'source': reference_paper if reference_paper else 'Unknown',
                'page': 1
            }
        )
        evidence_list.append(evidence)
        documents.append(doc)
    else:
        # Use the provided chunk text
        chunk_id = f"test_chunk_{question_no}"
        evidence = Evidence(
            chunk_id=chunk_id,
            summary=related_chunks.strip(),
            score=9,  # High score since it's the relevant chunk
            citation=reference_paper if reference_paper else "Unknown Source"
        )
        doc = Document(
            page_content=related_chunks.strip(),
            metadata={
                'chunk_id': chunk_id,
                'source': reference_paper if reference_paper else 'Unknown',
                'page': 1
            }
        )
        evidence_list.append(evidence)
        documents.append(doc)
    
    return evidence_list, documents


def normalize_answer(answer: str) -> Set[str]:
    """Normalize answer string to a set of strings for comparison."""
    if not answer:
        return set()
    
    answer = answer.strip()
    
    # Try to parse as JSON array directly (CSV already has JSON format)
    try:
        parsed = parse_json_array(answer)
        return {str(item).strip().lower() for item in parsed}
    except:
        # Fallback: treat as single string or comma-separated
        if ',' in answer:
            items = [item.strip().strip('"').strip("'") for item in answer.split(',')]
            return {item.lower() for item in items if item}
        else:
            return {answer.lower()}


def format_question_with_options(question: str, wrong_options: List[str], correct_answers: List[str]) -> tuple[str, Dict[str, str], Set[str]]:
    """
    Format question with all options (wrong + correct) and return option mapping.
    
    Returns:
        formatted_question: The question with all options listed
        option_map: Mapping from letter (A, B, C...) to option text
        correct_letters: Set of letters that are correct answers
    """
    # Combine all individual options into one list
    all_options = wrong_options + correct_answers
    option_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
    
    # Create mapping from letter to full option text
    option_map = {}
    options_text_list = []
    correct_letters = set()
    
    for i, opt in enumerate(all_options):
        if i < len(option_letters):
            letter = option_letters[i]
            option_map[letter] = opt
            options_text_list.append(f"{letter}. {opt}")
            
            # Mark which letters correspond to correct answers
            if opt in correct_answers:
                correct_letters.add(letter)
    
    options_text = "\n".join(options_text_list)
    formatted_question = f"{question}\n\nSelect ALL correct answer(s) from the following options:\n{options_text}"
    
    return formatted_question, option_map, correct_letters


def evaluate_answer(predicted: List[str], correct: Set[str]) -> Dict[str, any]:
    """Evaluate a single answer prediction."""
    predicted_set = {str(item).strip().lower() for item in predicted}
    correct_set = correct
    
    # Check for exact match
    is_exact_match = predicted_set == correct_set
    
    # Check for partial match (all predicted are correct, or all correct are predicted)
    all_predicted_correct = predicted_set.issubset(correct_set) if predicted_set else False
    all_correct_predicted = correct_set.issubset(predicted_set) if correct_set else False
    
    # Calculate precision and recall
    if predicted_set:
        precision = len(predicted_set & correct_set) / len(predicted_set)
    else:
        precision = 0.0
    
    if correct_set:
        recall = len(predicted_set & correct_set) / len(correct_set)
    else:
        recall = 0.0
    
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'is_correct': 1 if is_exact_match else 0,
        'is_partial': 1 if (all_predicted_correct or all_correct_predicted) and not is_exact_match else 0,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predicted': list(predicted_set),
        'correct': list(correct_set)
    }


def load_test_data(csv_path: str) -> List[Dict]:
    """Load test data from CSV file."""
    test_data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip sample rows
            if row.get('Domain', '').lower() == 'sample' or row.get('Question No.', '').lower() == 'sample':
                continue
            # Skip empty rows
            if not row.get('Question', '').strip():
                continue
            test_data.append(row)
    return test_data


def run_evaluation(csv_path: str, max_questions: int = None, debug: bool = False):
    """Run evaluation on the test dataset."""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("Answer LLM Evaluation")
    logger.info("=" * 80)
    
    # Load test data
    logger.info(f"Loading test data from {csv_path}...")
    test_data = load_test_data(csv_path)
    if max_questions:
        test_data = test_data[:max_questions]
    logger.info(f"Loaded {len(test_data)} test questions.")
    
    # Initialize Answer LLM
    logger.info("Initializing Answer LLM...")
    answer_tool = AnswerLLMTool()
    logger.info("Answer LLM initialized.")
    
    # Evaluation results
    results = []
    total = 0
    correct = 0
    partial = 0
    incorrect = 0
    
    # Process each question with progress bar
    pbar = tqdm(test_data, desc="Evaluating", unit="question")
    for idx, row in enumerate(pbar, 1):
        question_no = row.get('Question No.', str(idx))
        question = row.get('Question', '').strip()
        wrong_options_str = row.get('Wrong Options', '[]')
        answer_str = row.get('Answer', '[]')
        reference_paper = row.get('Reference Paper(s)', '')
        related_chunks = row.get('Related Chunk(s)', '')
        background_context = row.get('Background Context', '')
        
        if not question:
            continue
        
        # Update progress bar description with current stats
        pbar.set_description(f"Evaluating [✓{correct} ⚠{partial} ✗{incorrect}]")
        
        logger.info(f"[{idx}/{len(test_data)}] Processing Question {question_no}...")
        logger.info(f"Question: {question[:100]}...")
        
        # Parse options
        logger.debug(f"Raw wrong_options_str: {wrong_options_str}")
        logger.debug(f"Raw answer_str: {answer_str}")
        
        wrong_options = parse_json_array(wrong_options_str)
        correct_answers = parse_json_array(answer_str)
        
        logger.debug(f"Parsed wrong_options: {wrong_options} (type: {type(wrong_options)}, len: {len(wrong_options)})")
        logger.debug(f"Parsed correct_answers: {correct_answers} (type: {type(correct_answers)}, len: {len(correct_answers)})")
        
        if not correct_answers:
            logger.warning(f"Skipping: No correct answers provided")
            continue
        
        # Format question with options and get option mapping
        formatted_question, option_map, correct_letters = format_question_with_options(question, wrong_options, correct_answers)
        
        # Log all options provided to LLM
        logger.info(f"Options provided to LLM ({len(option_map)} total, {len(correct_letters)} correct):")
        for letter, option_text in sorted(option_map.items()):
            is_correct = letter in correct_letters
            marker = "✓ CORRECT" if is_correct else "  WRONG"
            logger.info(f"  {letter}. {option_text[:80]}{'...' if len(option_text) > 80 else ''} [{marker}]")
        
        # Create mock evidence and documents
        evidence_list, documents = create_mock_evidence_and_documents(
            related_chunks, reference_paper, question_no
        )
        
        # Create background
        background = Background(
            question=question,
            background_text=background_context if background_context else "No background context provided."
        )
        
        # Get prediction
        try:
            predicted_answer = answer_tool.answer(
                evidence_list=evidence_list,
                background=background,
                documents=documents,
                question=formatted_question,
                if_debug=debug
            )
            
            predicted_options = predicted_answer.answers
            if not predicted_options or (len(predicted_options) == 1 and predicted_options[0].lower() in ['i cannot answer', "i can't answer"]):
                predicted_options = []
            
            # Map option letters to full text
            predicted_full_text = []
            for opt in predicted_options:
                opt_str = str(opt).strip().upper()
                if opt_str in option_map:
                    predicted_full_text.append(option_map[opt_str])
                else:
                    # If it's already full text or unknown, keep as is
                    predicted_full_text.append(opt)
            
            predicted_options = predicted_full_text
            
            # Evaluate: compare predicted option texts with the correct_answers list
            # Convert correct_answers to a set for comparison
            correct_set = {opt.lower() for opt in correct_answers}
            eval_result = evaluate_answer(predicted_options, correct_set)
            
            total += 1
            if eval_result['is_correct']:
                correct += 1
                status = "CORRECT"
            elif eval_result['is_partial']:
                partial += 1
                status = "PARTIAL"
            else:
                incorrect += 1
                status = "INCORRECT"
            
            logger.info(f"  {status}")
            logger.info(f"  Predicted ({len(predicted_options)} option(s)):")
            for pred in predicted_options:
                logger.info(f"    - {pred}")
            logger.info(f"  Correct ({len(correct_set)} option(s)):")
            for cor in sorted(correct_set):
                logger.info(f"    - {cor}")
            logger.info(f"  Metrics: Precision={eval_result['precision']:.2f}, Recall={eval_result['recall']:.2f}, F1={eval_result['f1']:.2f}")
            
            # Update progress bar with current stats
            pbar.set_description(f"Evaluating [✓{correct} ⚠{partial} ✗{incorrect}]")
            
            results.append({
                'question_no': question_no,
                'question': question,
                'predicted': predicted_options,
                'correct': list(correct_set),
                'is_correct': eval_result['is_correct'],
                'is_partial': eval_result['is_partial'],
                'precision': eval_result['precision'],
                'recall': eval_result['recall'],
                'f1': eval_result['f1'],
                'confidence': predicted_answer.confidence
            })
            
        except Exception as e:
            logger.error(f"ERROR: {str(e)}")
            if debug:
                import traceback
                logger.error(traceback.format_exc())
            incorrect += 1
            total += 1
            pbar.set_description(f"Evaluating [✓{correct} ⚠{partial} ✗{incorrect}]")
            results.append({
                'question_no': question_no,
                'question': question,
                'predicted': [],
                'correct': [opt.lower() for opt in correct_answers],
                'is_correct': 0,
                'is_partial': 0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'confidence': 0.0,
                'error': str(e)
            })
    
    # Calculate final metrics
    logger.info("=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total Questions:     {total}")
    logger.info(f"Correct (Exact):     {correct} ({correct/total*100:.1f}%)")
    logger.info(f"Partial Match:       {partial} ({partial/total*100:.1f}%)")
    logger.info(f"Incorrect:           {incorrect} ({incorrect/total*100:.1f}%)")
    
    if results:
        avg_precision = sum(r['precision'] for r in results) / len(results)
        avg_recall = sum(r['recall'] for r in results) / len(results)
        avg_f1 = sum(r['f1'] for r in results) / len(results)
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        
        logger.info(f"Average Precision:   {avg_precision:.3f}")
        logger.info(f"Average Recall:      {avg_recall:.3f}")
        logger.info(f"Average F1 Score:    {avg_f1:.3f}")
        logger.info(f"Average Confidence:  {avg_confidence:.3f}")
    
    logger.info("=" * 80)
    
    # Save detailed results
    output_file = Path(__file__).parent / "answer_llm_evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total': total,
                'correct': correct,
                'partial': partial,
                'incorrect': incorrect,
                'accuracy': correct/total if total > 0 else 0.0,
                'avg_precision': avg_precision if results else 0.0,
                'avg_recall': avg_recall if results else 0.0,
                'avg_f1': avg_f1 if results else 0.0,
                'avg_confidence': avg_confidence if results else 0.0
            },
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Detailed results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Answer LLM on golden test data")
    parser.add_argument(
        "--csv",
        type=str,
        default="data/PaperQA Golden Test Data.csv",
        help="Path to CSV test data file (relative to project root or absolute)"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions to evaluate (for testing)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    
    args = parser.parse_args()
    
    # Setup logging with debug flag
    logger = setup_logging(debug=args.debug)
    
    # Resolve CSV path
    if os.path.isabs(args.csv):
        csv_path = Path(args.csv)
    else:
        # Resolve relative to project root
        csv_path = project_root / args.csv
    
    if not csv_path.exists():
        logger.error(f"CSV file not found at {csv_path}")
        sys.exit(1)
    
    run_evaluation(str(csv_path), args.max_questions, args.debug)

