# VQA Medical

Visual Question Answering for Medical Images using the ImageCLEF 2019 VQA-Med dataset.

## Project Structure

```
vqa_med/
├── vqamed/                    # Main package
│   ├── __init__.py           # Package exports
│   ├── config.py             # Configuration dataclass
│   ├── dataset.py            # VQADataset & data parsing
│   ├── encoders.py           # VisualEncoder (DenseNet121), ViTEncoder & TextEncoder (BERT)
│   ├── fusion.py             # CrossAttentionFusion module
│   ├── decoder.py            # AnswerDecoder (Transformer)
│   ├── model.py              # VQAModel (complete pipeline)
│   ├── training.py           # Training & validation functions
│   └── visualization.py      # Loss plotting utilities
├── train.py                  # Main training script
├── pyproject.toml            # Project dependencies
└── README.md
```

## Installation

```bash
uv sync
```

## Training

### Quick Start

```bash
# Train with default settings (DenseNet121 encoder)
python train.py

# Train with Vision Transformer encoder
python train.py --visual-encoder vit
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--train-path` | str | `data/ImageClef-2019-VQA-Med/Training` | Path to training data |
| `--val-path` | str | `data/ImageClef-2019-VQA-Med/Validation` | Path to validation data |
| `--batch-size` | int | `32` | Batch size for training |
| `--epochs` | int | `30` | Number of training epochs |
| `--lr` | float | `1e-4` | Learning rate |
| `--patience` | int | `5` | Early stopping patience |
| `--save-path` | str | `checkpoints/best_model.pt` | Path to save best model |
| `--visual-encoder` | str | `densenet` | Visual encoder: `densenet` or `vit` |

### Training Examples

**Basic training with DenseNet121:**
```bash
python train.py \
    --batch-size 32 \
    --epochs 30 \
    --lr 1e-4
```

**Training with Vision Transformer:**
```bash
python train.py \
    --visual-encoder vit \
    --batch-size 16 \
    --epochs 50 \
    --lr 5e-5
```

**Custom paths and early stopping:**
```bash
python train.py \
    --train-path /path/to/train \
    --val-path /path/to/val \
    --save-path models/my_model.pt \
    --patience 10
```

### Training Output

During training, the script will:
1. Display train/validation loss per epoch
2. Save the best model (lowest validation loss) to `--save-path`
3. Apply early stopping if validation loss doesn't improve for `--patience` epochs
4. Generate a loss plot (`loss_plot.png`) at the end of training

### Using as a Library

```python
from vqamed import Config, VQAModel, VQADataset

config = Config(
    train_path="data/Training",
    batch_size=32,
    num_epochs=30,
    visual_encoder="vit"  # or "densenet"
)

model = VQAModel.from_config(config, vocab_size=30522)
```

## Architecture

1. **VisualEncoder**: DenseNet121 or ViT-B/16 backbone → 512-dim embeddings
2. **TextEncoder**: BERT-base → 512-dim embeddings  
3. **CrossAttentionFusion**: 8-head cross-attention for multimodal fusion
4. **AnswerDecoder**: 4-layer Transformer decoder for answer generation

### Visual Encoders

| Encoder | Backbone | Parameters | Input Size |
|---------|----------|------------|------------|
| `densenet` | DenseNet121 | ~7M | 224×224 |
| `vit` | ViT-B/16 | ~86M | 224×224 |

## Dataset

Download the [ImageCLEF 2019 VQA-Med dataset](https://www.imageclef.org/2019/medical/vqa) and organize as:

```
data/ImageClef-2019-VQA-Med/
├── Training/
│   ├── images/
│   └── all_qa_pairs.txt
├── Validation/
│   ├── images/
│   └── all_qa_pairs.txt
└── Test/
    ├── images/
    └── questions_w_ref_answers.txt
```
