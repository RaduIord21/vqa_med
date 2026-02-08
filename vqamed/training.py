"""Training utilities for VQA Medical."""

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    tokenizer: Any,
    device: torch.device,
    max_len: int = 32,
    log_interval: int = 20
) -> float:
    """
    Train model for one epoch.
    
    Args:
        model: VQA model to train.
        dataloader: Training data loader.
        optimizer: Optimizer.
        tokenizer: Tokenizer for answers.
        device: Training device.
        max_len: Maximum sequence length.
        log_interval: Logging interval (batches).
        
    Returns:
        Average training loss.
    """
    model.train()
    total_loss = 0.0

    for cnt, batch in enumerate(dataloader):
        if cnt % log_interval == 0:
            print(f'{cnt} / {len(dataloader)}')
            
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        answers = batch['answer']

        # Tokenize answers
        answer_tokens = tokenizer(
            answers,
            padding='max_length',
            truncation=True,
            max_length=max_len,
            return_tensors='pt'
        )
        decoder_input_ids = answer_tokens['input_ids'][:, :-1].to(device)
        labels = answer_tokens['input_ids'][:, 1:].to(device)

        # Forward pass
        logits = model(
            image=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids
        )

        # Compute loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=tokenizer.pad_token_id
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def validate_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    tokenizer: Any,
    device: torch.device,
    max_len: int = 32,
    log_interval: int = 20
) -> float:
    """
    Validate model for one epoch.
    
    Args:
        model: VQA model to validate.
        dataloader: Validation data loader.
        tokenizer: Tokenizer for answers.
        device: Device.
        max_len: Maximum sequence length.
        log_interval: Logging interval (batches).
        
    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0

    for cnt, batch in enumerate(dataloader):
        if cnt % log_interval == 0:
            print(f"{cnt} / {len(dataloader)}")
            
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        answers = batch['answer']

        answer_tokens = tokenizer(
            answers,
            padding='max_length',
            truncation=True,
            max_length=max_len,
            return_tensors='pt'
        )

        decoder_input_ids = answer_tokens['input_ids'][:, :-1].to(device)
        labels = answer_tokens['input_ids'][:, 1:].to(device)

        logits = model(images, input_ids, attention_mask, decoder_input_ids)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=tokenizer.pad_token_id
        )
        total_loss += loss.item()

    return total_loss / len(dataloader)


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss.
            
        Returns:
            True if model improved, False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False
