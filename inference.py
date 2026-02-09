#!/usr/bin/env python
"""Inference module for VQA Medical models - generate predictions on new images."""

import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import textwrap

import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from vqamed import Config, VQAModel


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference on medical images with VQA model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image inference
  python inference.py --checkpoint model.pt --image scan.jpg --question "What is the modality?"
  
  # Batch inference from folder
  python inference.py --checkpoint model.pt --images-dir ./test_images --questions-file questions.txt
  
  # Interactive mode
  python inference.py --checkpoint model.pt --interactive
        """
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--visual-encoder",
        type=str,
        choices=["densenet", "vit"],
        default="densenet",
        help="Visual encoder type (must match checkpoint)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single image for inference"
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question about the image"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Directory with images for batch inference"
    )
    parser.add_argument(
        "--questions-file",
        type=str,
        default=None,
        help="File with questions (format: image_name|question per line)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
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
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of samples to show in batch mode (use -1 for all)"
    )
    parser.add_argument(
        "--save-visual",
        type=str,
        default=None,
        help="Save visual grid of results as JPG (e.g., results_densenet.jpg)"
    )
    return parser.parse_args()


class VQAInference:
    """Inference wrapper for VQA Medical models."""
    
    def __init__(
        self,
        checkpoint_path: str,
        visual_encoder: str = "densenet",
        max_len: int = 32,
        device: Optional[str] = None
    ):
        """
        Initialize inference module.
        
        Args:
            checkpoint_path: Path to model checkpoint.
            visual_encoder: Type of visual encoder ("densenet" or "vit").
            max_len: Maximum generation length.
            device: Device to use (auto-detect if None).
        """
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_len = max_len
        
        print(f"Using device: {self.device}")
        print(f"Loading model from: {checkpoint_path}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer.pad_token = '[PAD]'
        
        # Initialize model
        config = Config(visual_encoder=visual_encoder, max_len=max_len)
        self.model = VQAModel.from_config(config, vocab_size=len(self.tokenizer))
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print("Model loaded successfully!\n")
    
    @torch.no_grad()
    def predict(self, image_path: str, question: str) -> str:
        """
        Generate answer for a single image-question pair.
        
        Args:
            image_path: Path to the image.
            question: Question about the image.
            
        Returns:
            Generated answer string.
        """
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Tokenize question
        tokens = self.tokenizer(
            question,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)
        
        # Generate answer
        decoder_input_ids = torch.tensor(
            [[self.tokenizer.cls_token_id]], device=self.device
        )
        
        for _ in range(self.max_len):
            logits = self.model(
                image=image_tensor,
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids
            )
            
            # Greedy decoding
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
            
            # Stop at EOS
            if next_token.item() == self.tokenizer.sep_token_id:
                break
        
        # Decode answer
        generated_ids = decoder_input_ids[0, 1:].tolist()
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return answer
    
    def predict_batch(
        self,
        samples: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Generate answers for multiple image-question pairs.
        
        Args:
            samples: List of dicts with 'image_path' and 'question' keys.
            
        Returns:
            List of dicts with added 'answer' key.
        """
        results = []
        for i, sample in enumerate(samples):
            try:
                answer = self.predict(sample['image_path'], sample['question'])
                results.append({
                    'image_path': sample['image_path'],
                    'question': sample['question'],
                    'answer': answer
                })
            except Exception as e:
                results.append({
                    'image_path': sample['image_path'],
                    'question': sample['question'],
                    'answer': f"[ERROR: {str(e)}]"
                })
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(samples)} samples...")
        
        return results


def print_result(image_path: str, question: str, answer: str) -> None:
    """Pretty print a single result."""
    print("-" * 60)
    print(f"📷 Image:    {image_path}")
    print(f"❓ Question: {question}")
    print(f"💡 Answer:   {answer}")
    print("-" * 60)


def save_visual_grid(
    results: List[Dict[str, str]],
    output_path: str,
    model_name: str = "Model",
    num_samples: int = 4
) -> None:
    """
    Save a visual grid of images with questions and answers.
    
    Args:
        results: List of dicts with 'image_path', 'question', 'answer' keys.
        output_path: Path to save the JPG file.
        model_name: Name of the model for the title.
        num_samples: Number of samples to display (default 4 for 2x2 grid).
    """
    # Take only the first num_samples
    samples = results[:num_samples]
    n = len(samples)
    
    if n == 0:
        print("No results to visualize!")
        return
    
    # Calculate grid dimensions
    cols = 2
    rows = (n + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 7 * rows))
    fig.suptitle(f'VQA Medical Results - {model_name}', fontsize=16, fontweight='bold')
    
    # Flatten axes for easy iteration
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    
    for idx, sample in enumerate(samples):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col]
        
        # Load and display image
        try:
            img = Image.open(sample['image_path']).convert('RGB')
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading image:\n{e}", 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.axis('off')
        
        # Wrap text for better display
        question = sample['question']
        answer = sample['answer']
        image_name = Path(sample['image_path']).name
        
        # Wrap long text
        wrapped_q = textwrap.fill(f"Q: {question}", width=50)
        wrapped_a = textwrap.fill(f"A: {answer}", width=50)
        
        title = f"{image_name}\n{wrapped_q}\n{wrapped_a}"
        ax.set_title(title, fontsize=10, pad=10, loc='center')
    
    # Hide empty subplots
    for idx in range(n, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row][col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\nVisual results saved to: {output_path}")


def load_questions_file(filepath: str, images_dir: str) -> List[Dict[str, str]]:
    """
    Load questions from file.
    
    Expected format: image_name|question (one per line)
    """
    samples = []
    images_dir = Path(images_dir)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('|')
            if len(parts) >= 2:
                image_name = parts[0].strip()
                question = parts[1].strip()
                
                # Try different extensions
                image_path = None
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    candidate = images_dir / f"{image_name}{ext}"
                    if candidate.exists():
                        image_path = str(candidate)
                        break
                    # Also try with extension already in name
                    candidate = images_dir / image_name
                    if candidate.exists():
                        image_path = str(candidate)
                        break
                
                if image_path:
                    samples.append({
                        'image_path': image_path,
                        'question': question
                    })
                else:
                    print(f"Warning: Image not found for '{image_name}'")
    
    return samples


def interactive_mode(inference: VQAInference) -> None:
    """Run interactive inference session."""
    print("\n" + "=" * 60)
    print("        VQA Medical - Interactive Mode")
    print("=" * 60)
    print("Enter image path and question, or 'quit' to exit.\n")
    
    while True:
        # Get image path
        image_path = input("Image path (or 'quit'): ").strip()
        if image_path.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not Path(image_path).exists():
            print(f"Error: Image not found at '{image_path}'\n")
            continue
        
        # Get question
        question = input("Question: ").strip()
        if not question:
            print("Error: Question cannot be empty\n")
            continue
        
        # Generate answer
        print("\nGenerating answer...")
        answer = inference.predict(image_path, question)
        print_result(image_path, question, answer)
        print()


def main():
    """Main inference function."""
    args = parse_args()
    
    # Initialize inference module
    inference = VQAInference(
        checkpoint_path=args.checkpoint,
        visual_encoder=args.visual_encoder,
        max_len=args.max_len
    )
    
    results = []
    
    # Mode 1: Interactive
    if args.interactive:
        interactive_mode(inference)
        return
    
    # Mode 2: Single image
    if args.image and args.question:
        answer = inference.predict(args.image, args.question)
        print_result(args.image, args.question, answer)
        results.append({
            'image_path': args.image,
            'question': args.question,
            'answer': answer
        })
    
    # Mode 3: Batch from questions file
    elif args.images_dir and args.questions_file:
        print(f"Loading questions from: {args.questions_file}")
        samples = load_questions_file(args.questions_file, args.images_dir)
        print(f"Found {len(samples)} valid image-question pairs\n")
        
        if not samples:
            print("No valid samples found!")
            return
        
        results = inference.predict_batch(samples)
        
        # Print results
        num_to_show = len(results) if args.num_samples == -1 else min(args.num_samples, len(results))
        print(f"\nShowing {num_to_show} of {len(results)} results:\n")
        
        for result in results[:num_to_show]:
            print_result(result['image_path'], result['question'], result['answer'])
    
    # Mode 4: Just images directory (use default questions)
    elif args.images_dir:
        images_dir = Path(args.images_dir)
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        if not image_files:
            print(f"No images found in {args.images_dir}")
            return
        
        # Default medical questions
        default_questions = [
            "What is the imaging modality?",
            "What organ is shown?",
            "Are there any abnormalities?",
        ]
        
        print(f"Found {len(image_files)} images")
        print(f"Using default questions: {default_questions}\n")
        
        num_images = len(image_files) if args.num_samples == -1 else min(args.num_samples, len(image_files))
        
        for img_path in image_files[:num_images]:
            print(f"\n{'='*60}")
            print(f"Image: {img_path.name}")
            print('='*60)
            
            for question in default_questions:
                answer = inference.predict(str(img_path), question)
                print(f"Q: {question}")
                print(f"A: {answer}\n")
                
                results.append({
                    'image_path': str(img_path),
                    'question': question,
                    'answer': answer
                })
    
    else:
        print("Error: Please specify one of:")
        print("  --image and --question (single inference)")
        print("  --images-dir and --questions-file (batch inference)")
        print("  --images-dir (batch with default questions)")
        print("  --interactive (interactive mode)")
        return
    
    # Save results if requested
    if args.output and results:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")
    
    # Save visual grid if requested
    if args.save_visual and results:
        save_visual_grid(
            results=results,
            output_path=args.save_visual,
            model_name=args.visual_encoder.upper(),
            num_samples=args.num_samples if args.num_samples > 0 else 4
        )


if __name__ == "__main__":
    main()
