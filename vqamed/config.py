"""Configuration for VQA Medical model."""

from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class Config:
    """Configuration class for VQA Medical training."""
    
    # Paths
    train_path: str = "data/Training"
    validation_path: str = "data/Validation"
    test_path: str = "data/Test"
    save_path: str = "checkpoints/best_model.pt"
    
    # Model
    embed_dim: int = 512
    num_heads: int = 8
    decoder_layers: int = 4
    max_len: int = 32
    text_model_name: str = "bert-base-uncased"
    visual_encoder: str = "densenet"  # "densenet" or "vit"
    
    # Training
    batch_size: int = 32
    num_epochs: int = 30
    learning_rate: float = 1e-4
    patience: int = 5
    
    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
    
    @property
    def train_images_dir(self) -> str:
        return f"{self.train_path}/images"
    
    @property
    def train_qa_file(self) -> str:
        return f"{self.train_path}/all_qa_pairs.txt"
    
    @property
    def val_images_dir(self) -> str:
        return f"{self.validation_path}/images"
    
    @property
    def val_qa_file(self) -> str:
        return f"{self.validation_path}/all_qa_pairs.txt"
    
    @property
    def test_images_dir(self) -> str:
        return f"{self.test_path}/images"
    
    @property
    def test_qa_file(self) -> str:
        return f"{self.test_path}/questions_w_ref_answers.txt"
