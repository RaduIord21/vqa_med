#!/usr/bin/env python
"""Main training script for VQA Medical."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from vqamed import (
    Config,
    VQADataset,
    VQAModel,
)
from vqamed.training import train_epoch, validate_epoch, EarlyStopping
from vqamed.visualization import plot_losses

root_dataset = "data/ImageClef-2019-VQA-Med"
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train VQA Medical model")
    parser.add_argument(
        "--train-path",
        type=str,
        default=f"{root_dataset}/Training",
        help="Path to training data"
    )
    parser.add_argument(
        "--val-path",
        type=str,
        default=f"{root_dataset}/Validation",
        help="Path to validation data"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="checkpoints/best_model.pt",
        help="Model save path"
    )
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Create config
    config = Config(
        train_path=args.train_path,
        validation_path=args.val_path,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        save_path=args.save_path,
    )
    
    device = torch.device(config.device)
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
    tokenizer.pad_token = '[PAD]'
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = VQADataset(
        images_dir=config.train_images_dir,
        qa_file=config.train_qa_file,
        tokenizer=tokenizer,
        max_len=config.max_len
    )
    
    val_dataset = VQADataset(
        images_dir=config.val_images_dir,
        qa_file=config.val_qa_file,
        tokenizer=tokenizer,
        max_len=config.max_len
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    print("Initializing model...")
    model = VQAModel.from_config(config, vocab_size=len(tokenizer))
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.patience)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    print(f"\nStarting training for {config.num_epochs} epochs...")
    print("-" * 50)
    
    for epoch in tqdm(range(config.num_epochs), desc="Training"):
        train_loss = train_epoch(
            model, train_loader, optimizer, tokenizer, device, config.max_len
        )
        val_loss = validate_epoch(
            model, val_loader, tokenizer, device, config.max_len
        )
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"\nEpoch {epoch + 1}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")
        
        # Early stopping check
        improved = early_stopping(val_loss)
        if improved:
            Path(config.save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), config.save_path)
            print("✅ Model improved. Saving.")
        else:
            print(f"⚠️ No improvement. Patience: {early_stopping.counter}/{config.patience}")
        
        if early_stopping.should_stop:
            print("🛑 Early stopping triggered.")
            break
    
    # Plot losses
    plot_losses(train_losses, val_losses, "loss_plot.png")
    
    print(f"\nTraining complete! Best model saved to: {config.save_path}")


if __name__ == "__main__":
    main()
