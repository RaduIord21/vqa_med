"""Evaluation metrics for VQA Medical."""

from collections import Counter
from typing import List, Dict, Any
import re


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.
    
    Args:
        answer: Raw answer string.
        
    Returns:
        Normalized answer (lowercase, stripped, no punctuation).
    """
    answer = answer.lower().strip()
    # Remove punctuation
    answer = re.sub(r'[^\w\s]', '', answer)
    # Normalize whitespace
    answer = ' '.join(answer.split())
    return answer


def exact_match(predictions: List[str], references: List[str]) -> float:
    """
    Calculate exact match accuracy.
    
    Args:
        predictions: List of predicted answers.
        references: List of ground truth answers.
        
    Returns:
        Exact match accuracy (0-1).
    """
    correct = 0
    for pred, ref in zip(predictions, references):
        if normalize_answer(pred) == normalize_answer(ref):
            correct += 1
    return correct / len(predictions) if predictions else 0.0


def token_f1(prediction: str, reference: str) -> float:
    """
    Calculate token-level F1 score for a single prediction.
    
    Args:
        prediction: Predicted answer.
        reference: Ground truth answer.
        
    Returns:
        F1 score (0-1).
    """
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1


def average_f1(predictions: List[str], references: List[str]) -> float:
    """
    Calculate average token-level F1 score.
    
    Args:
        predictions: List of predicted answers.
        references: List of ground truth answers.
        
    Returns:
        Average F1 score (0-1).
    """
    if not predictions:
        return 0.0
    
    f1_scores = [token_f1(p, r) for p, r in zip(predictions, references)]
    return sum(f1_scores) / len(f1_scores)


def bleu_score(predictions: List[str], references: List[str], max_n: int = 4) -> Dict[str, float]:
    """
    Calculate BLEU scores (1-4 gram).
    
    Args:
        predictions: List of predicted answers.
        references: List of ground truth answers.
        max_n: Maximum n-gram to calculate.
        
    Returns:
        Dictionary with BLEU-1, BLEU-2, etc.
    """
    from collections import Counter
    import math
    
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    scores = {}
    
    for n in range(1, max_n + 1):
        total_matches = 0
        total_count = 0
        
        for pred, ref in zip(predictions, references):
            pred_tokens = normalize_answer(pred).split()
            ref_tokens = normalize_answer(ref).split()
            
            if len(pred_tokens) < n:
                continue
                
            pred_ngrams = get_ngrams(pred_tokens, n)
            ref_ngrams = get_ngrams(ref_tokens, n)
            
            matches = sum((pred_ngrams & ref_ngrams).values())
            total_matches += matches
            total_count += sum(pred_ngrams.values())
        
        scores[f'bleu_{n}'] = total_matches / total_count if total_count > 0 else 0.0
    
    return scores


def compute_all_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        predictions: List of predicted answers.
        references: List of ground truth answers.
        
    Returns:
        Dictionary with all metrics.
    """
    metrics = {
        'exact_match': exact_match(predictions, references),
        'f1': average_f1(predictions, references),
    }
    metrics.update(bleu_score(predictions, references))
    
    return metrics


def print_metrics(metrics: Dict[str, float], model_name: str = "Model") -> None:
    """
    Pretty print metrics.
    
    Args:
        metrics: Dictionary of metrics.
        model_name: Name of the model for display.
    """
    print(f"\n{'='*50}")
    print(f"  {model_name} - Evaluation Results")
    print(f"{'='*50}")
    print(f"  Exact Match:  {metrics['exact_match']*100:.2f}%")
    print(f"  F1 Score:     {metrics['f1']*100:.2f}%")
    print(f"  BLEU-1:       {metrics['bleu_1']*100:.2f}%")
    print(f"  BLEU-2:       {metrics['bleu_2']*100:.2f}%")
    print(f"  BLEU-3:       {metrics['bleu_3']*100:.2f}%")
    print(f"  BLEU-4:       {metrics['bleu_4']*100:.2f}%")
    print(f"{'='*50}\n")
