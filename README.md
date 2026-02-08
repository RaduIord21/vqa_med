# VQA Medical

Visual Question Answering for Medical Images using the ImageCLEF 2019 VQA-Med dataset.

## Project Structure

```
vqa_med/
├── vqamed/                    # Main package
│   ├── __init__.py           # Package exports
│   ├── config.py             # Configuration dataclass
│   ├── dataset.py            # VQADataset & data parsing
│   ├── encoders.py           # VisualEncoder (DenseNet121) & TextEncoder (BERT)
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

## Usage

### Training

```bash
python train.py \
    --train-path data/Training \
    --val-path data/Validation \
    --batch-size 32 \
    --epochs 30 \
    --lr 1e-4
```

### Using as a library

```python
from vqamed import Config, VQAModel, VQADataset

config = Config(
    train_path="data/Training",
    batch_size=32,
    num_epochs=30
)

model = VQAModel.from_config(config, vocab_size=30522)
```

## Architecture

1. **VisualEncoder**: DenseNet121 backbone → 512-dim embeddings
2. **TextEncoder**: BERT-base → 512-dim embeddings  
3. **CrossAttentionFusion**: 8-head cross-attention for multimodal fusion
4. **AnswerDecoder**: 4-layer Transformer decoder for answer generation

## Dataset

Download the [ImageCLEF 2019 VQA-Med dataset](https://www.imageclef.org/2019/medical/vqa) and organize as:

```
data/
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
