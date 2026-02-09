#!/usr/bin/env python
"""Evaluation script for VQA Medical models."""

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from vqamed import Config, VQADataset, VQAModel
from vqamed.metrics import compute_all_metrics, print_metrics


root_dataset = "data/ImageClef-2019-VQA-Med"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate VQA Medical models")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=f"{root_dataset}/Test",
        help="Path to test data"
    )
    parser.add_argument(
        "--visual-encoder",
        type=str,
        choices=["densenet", "vit"],
        default="densenet",
        help="Visual encoder type used in the checkpoint"
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
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save predictions (optional)"
    )
    return parser.parse_args()


@torch.no_grad()
def generate_answer(
    model: torch.nn.Module,
    image: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    max_len: int = 32,
    device: torch.device = torch.device("cpu")
) -> str:
    """
    Generate answer using greedy decoding.
    
    Args:
        model: VQA model.
        image: Input image tensor [1, C, H, W].
        input_ids: Question token IDs [1, T].
        attention_mask: Question attention mask [1, T].
        tokenizer: Tokenizer.
        max_len: Maximum generation length.
        device: Device.
        
    Returns:
        Generated answer string.
    """
    model.eval()
    
    # Start with BOS token
    decoder_input_ids = torch.tensor([[tokenizer.cls_token_id]], device=device)
    
    for _ in range(max_len):
        logits = model(
            image=image,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids
        )
        
        # Get next token (greedy)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
        
        # Stop if EOS
        if next_token.item() == tokenizer.sep_token_id:
            break
    
    # Decode tokens to string
    generated_ids = decoder_input_ids[0, 1:].tolist()  # Skip BOS
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return answer


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    max_len: int = 32
) -> Tuple[List[str], List[str]]:
    """
    Evaluate model on dataset.
    
    Args:
        model: VQA model.
        dataloader: Test data loader.
        tokenizer: Tokenizer.
        device: Device.
        max_len: Maximum generation length.
        
    Returns:
        Tuple of (predictions, references).
    """
    model.eval()
    predictions = []
    references = []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        answers = batch['answer']
        
        # Generate answers one by one (batch generation is complex)
        for i in range(images.size(0)):
            pred = generate_answer(
                model=model,
                image=images[i:i+1],
                input_ids=input_ids[i:i+1],
                attention_mask=attention_mask[i:i+1],
                tokenizer=tokenizer,
                max_len=max_len,
                device=device
            )
            predictions.append(pred)
            references.append(answers[i])
    
    return predictions, references


def save_predictions(
    predictions: List[str],
    references: List[str],
    output_path: str
) -> None:
    """Save predictions to file."""
    with open(output_path, 'w') as f:
        f.write("prediction\treference\n")
        for pred, ref in zip(predictions, references):
            f.write(f"{pred}\t{ref}\n")
    print(f"Predictions saved to {output_path}")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create config
    config = Config(
        test_path=args.test_path,
        visual_encoder=args.visual_encoder,
        max_len=args.max_len
    )
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
    tokenizer.pad_token = '[PAD]'
    
    # Load test dataset
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
    
    # Create and load model
    print(f"Loading model from {args.checkpoint}...")
    print(f"Visual encoder: {args.visual_encoder}")
    
    model = VQAModel.from_config(config, vocab_size=len(tokenizer))
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    
    # Evaluate
    print("\nGenerating predictions...")
    predictions, references = evaluate_model(
        model=model,
        dataloader=test_loader,
        tokenizer=tokenizer,
        device=device,
        max_len=args.max_len
    )
    
    # Compute metrics
    metrics = compute_all_metrics(predictions, references)
    print_metrics(metrics, model_name=f"VQA-Med ({args.visual_encoder.upper()})")
    
    # Save predictions if requested
    if args.output:
        save_predictions(predictions, references, args.output)
    
    return metrics

    
if __name__ == "__main__":
    main()
