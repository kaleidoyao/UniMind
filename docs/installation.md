# UniMind Installation Guide

## Requirements

- Python 3.9
- CUDA 12.x
- 8× A100-80G (or equivalent) for full training; 1× GPU for evaluation

---

## 1. Create Conda Environment

Restore from the pinned environment file (includes all dependencies and the exact package versions used in the paper):

```bash
conda env create -f environment.yml
conda activate unimind
```

---

## 2. Install Flash Attention

Flash Attention must be built from source or installed via a pre-built wheel.

```bash
pip install flash-attn==2.3.6 --no-build-isolation
```

If the above fails (architecture mismatch), build from the bundled source:

```bash
cd InternVL-EEG/internvl_chat/flash-attention-2.3.6
pip install . --no-build-isolation
```

---

## 3. Download Pretrained Weights

### InternVL2 (LLM backbone)

Download the model weights from HuggingFace and place them under `InternVL-EEG/pretrained/`:

```bash
cd InternVL-EEG/pretrained

# Default backbone used in training (8B)
git clone https://huggingface.co/OpenGVLab/InternVL2-8B

# Smaller alternatives
git clone https://huggingface.co/OpenGVLab/InternVL2-1B
git clone https://huggingface.co/OpenGVLab/InternVL2-2B
git clone https://huggingface.co/OpenGVLab/InternVL2-4B
```

### LaBraM (EEG encoder)

Download LaBraM pretrained weights from the [LaBraM repository](https://github.com/935963004/LaBraM) and set the path in `get_models_v2()` inside:

```
InternVL-EEG/internvl_chat/internvl/model/LaBraM_main/labram_encoder_fin.py
```

---

## 4. Configure Dataset Paths

Edit the dataset config files directly with your local `.jsonl` paths:

- `InternVL-EEG/internvl_chat/shell/data/train_datasets.json` — set `annotation` for each dataset
- `InternVL-EEG/internvl_chat/shell/data/test_datasets.json` — set `train` and `test` for each dataset

Also fill in `ch_names`, `mean`, and `std` for each dataset (see [data_preparation.md](data_preparation.md) for how to compute them).

---
