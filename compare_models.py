#!/usr/bin/env python
"""Compare two VQA Medical models (DenseNet vs ViT)."""

import argparse
from typing import Dict

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from vqamed import Config, VQADataset, VQAModel
from vqamed.metrics import compute_all_metrics, print_metrics
from evaluate import evaluate_model


root_dataset = "data/ImageClef-2019-VQA-Med"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare VQA Medical models")
    parser.add_argument(
        "--densenet-checkpoint",
        type=str,
        required=True,
        help="Path to DenseNet model checkpoint"
    )
    parser.add_argument(
        "--vit-checkpoint",
        type=str,
        required=True,
        help="Path to ViT model checkpoint"
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=f"{root_dataset}/Test",
        help="Path to test data"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=32,
        help="Maximum sequence length"
    )
    return parser.parse_args()


def print_comparison(metrics_densenet: Dict[str, float], metrics_vit: Dict[str, float]) -> None:
    """Print side-by-side comparison of metrics."""
    print("\n" + "="*70)
    print("                    MODEL COMPARISON")
    print("="*70)
    print(f"{'Metric':<20} {'DenseNet':>15} {'ViT':>15} {'Δ (ViT-DenseNet)':>18}")
    print("-"*70)
    
    for metric in ['exact_match', 'f1', 'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4']:
        d_val = metrics_densenet[metric] * 100
        v_val = metrics_vit[metric] * 100
        delta = v_val - d_val
        delta_str = f"+{delta:.2f}%" if delta > 0 else f"{delta:.2f}%"
        
        metric_name = metric.replace('_', ' ').title()
        print(f"{metric_name:<20} {d_val:>14.2f}% {v_val:>14.2f}% {delta_str:>18}")
    
    print("="*70)
    
    # Summary
    vit_wins = sum(1 for m in metrics_densenet if metrics_vit[m] > metrics_densenet[m])
    total = len(metrics_densenet)
    
    if vit_wins > total / 2:
        print(f"\n✅ ViT is better on {vit_wins}/{total} metrics")
    elif vit_wins < total / 2:
        print(f"\n✅ DenseNet is better on {total - vit_wins}/{total} metrics")
    else:
        print(f"\n🔄 Models are tied ({vit_wins}/{total} metrics each)")


def main():
    """Main comparison function."""
    args = parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    config = Config(test_path=args.test_path, max_len=args.max_len)
    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
    tokenizer.pad_token = '[PAD]'
    
    # Load test dataset (once, shared between models)
    print("Loading test dataset...")
    test_dataset = VQADataset(
        images_dir=config.test_images_dir,
        qa_file=config.test_qa_file,
        tokenizer=tokenizer,
        max_len=config.max_len
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Evaluate DenseNet model
    print("\n" + "="*50)
    print("Evaluating DenseNet model...")
    print("="*50)
    
    config_densenet = Config(test_path=args.test_path, visual_encoder="densenet", max_len=args.max_len)
    model_densenet = VQAModel.from_config(config_densenet, vocab_size=len(tokenizer))
    model_densenet.load_state_dict(torch.load(args.densenet_checkpoint, map_location=device))
    model_densenet = model_densenet.to(device)
    
    preds_densenet, refs = evaluate_model(
        model=model_densenet,
        dataloader=test_loader,
        tokenizer=tokenizer,
        device=device,
        max_len=args.max_len
    )
    metrics_densenet = compute_all_metrics(preds_densenet, refs)
    print_metrics(metrics_densenet, model_name="DenseNet")
    
    # Free memory
    del model_densenet
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Evaluate ViT model
    print("\n" + "="*50)
    print("Evaluating ViT model...")
    print("="*50)
    
    config_vit = Config(test_path=args.test_path, visual_encoder="vit", max_len=args.max_len)
    model_vit = VQAModel.from_config(config_vit, vocab_size=len(tokenizer))
    model_vit.load_state_dict(torch.load(args.vit_checkpoint, map_location=device))
    model_vit = model_vit.to(device)
    
    preds_vit, _ = evaluate_model(
        model=model_vit,
        dataloader=test_loader,
        tokenizer=tokenizer,
        device=device,
        max_len=args.max_len
    )
    metrics_vit = compute_all_metrics(preds_vit, refs)
    print_metrics(metrics_vit, model_name="ViT")
    
    # Print comparison
    print_comparison(metrics_densenet, metrics_vit)


if __name__ == "__main__":
    main()
