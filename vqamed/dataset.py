"""Dataset module for VQA Medical."""

from pathlib import Path
from typing import Optional, Callable, Tuple, List, Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def parse_qa_set(qa_pairs_txt_path: str) -> Tuple[int, List[Dict[str, str]]]:
    """
    Parse QA pairs from a text file.
    
    Args:
        qa_pairs_txt_path: Path to the QA pairs text file.
        
    Returns:
        Tuple of (count, list of QA dictionaries).
    """
    qa_set = []

    with open(qa_pairs_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            elements = line.strip().split('|')
            if len(elements) > 3:
                image_id = elements[0]
                question = elements[2]
                answer = elements[3]
            else:
                image_id, question, answer = elements
            qa_set.append({
                'image_id': image_id,
                'question': question,
                'answer': answer
            })

    return len(qa_set), qa_set


class VQADataset(Dataset):
    """Dataset for Visual Question Answering on medical images."""
    
    def __init__(
        self,
        images_dir: str,
        qa_file: str,
        transform: Optional[Callable] = None,
        tokenizer: Optional[Any] = None,
        max_len: int = 32
    ):
        """
        Initialize VQA Dataset.
        
        Args:
            images_dir: Directory containing images (.jpg/.png).
            qa_file: Text file with QA pairs (format: image_id|...|question|answer).
            transform: Torchvision transforms for images.
            tokenizer: Tokenizer for questions.
            max_len: Maximum length for tokenization.
        """
        self.images_dir = Path(images_dir)
        self.length, self.items = parse_qa_set(qa_file)
        self.transform = transform or self._default_transform()
        self.tokenizer = tokenizer
        self.max_len = max_len

    @staticmethod
    def _default_transform() -> transforms.Compose:
        """Return default ImageNet-style transforms."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        img_path = self.images_dir / f"{item['image_id']}.jpg"
        
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        question = item['question']
        if self.tokenizer:
            tokens = self.tokenizer(
                question,
                padding='max_length',
                truncation=True,
                max_length=self.max_len,
                return_tensors='pt'
            )
            input_ids = tokens['input_ids'].squeeze(0)
            attention_mask = tokens['attention_mask'].squeeze(0)
        else:
            input_ids = question
            attention_mask = None

        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'answer': item['answer']
        }
