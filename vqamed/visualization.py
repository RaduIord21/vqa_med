"""Visualization utilities for VQA Medical."""

import matplotlib.pyplot as plt
from typing import List


def plot_losses(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str = "loss_plot.png"
) -> None:
    """
    Plot training and validation losses.
    
    Args:
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch.
        save_path: Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training vs Validation Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Loss plot saved to {save_path}")
