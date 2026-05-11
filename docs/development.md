# Development Guide

## Repository Layout

```
UniMind/
├── README.md
├── .gitignore
├── docs/                            ← documentation
│   ├── installation.md
│   ├── data_preparation.md
│   └── development.md               ← this file
├── demo_data/                       ← sample EEG pkl files + annotation jsonl
├── instruction_jsonl/               ← train/test JSONL files
└── InternVL-EEG/
    ├── internvl_chat/               ← main training & evaluation package
    │   ├── internvl/
    │   │   ├── model/
    │   │   │   ├── LaBraM_main/     ← EEG encoder (NeuralTransformer / LaBraM)
    │   │   │   └── internvl_chat/   ← InternVLChatModel + CTQFormer adapter
    │   │   ├── train/               ← dataset loader + finetune script
    │   │   └── patch/               ← monkey-patches for sampler / dataloader
    │   ├── eval/                    ← evaluation scripts
    │   └── shell/
    │       ├── train/train.sh       ← training launch script
    │       ├── evaluate/evaluate.sh ← evaluation launch script
    │       └── data/                ← dataset config JSONs (train_datasets.json, test_datasets.json)
    ├── pretrained/                  ← InternVL2 weights (not in git)
    └── work_dirs/                   ← training checkpoints (not in git)
```

## Training

```bash
cd InternVL-EEG/internvl_chat

# Step 1: fill in your local .jsonl paths in shell/data/train_datasets.json
# Step 2: launch training
MODEL_PATH=pretrained/InternVL2-8B bash shell/train/train.sh
```

## Evaluation

```bash
cd InternVL-EEG/internvl_chat
GPUS=1 bash shell/evaluate/evaluate.sh <checkpoint_path> <DATASET_NAME>
# DATASET_NAME: SEED | HMC | Workload | TUAB | TUEV | TUSL | SEEDIV | SHU | SleepEDF | SHHS
```

## Paths That Need Configuration

| What | How |
|---|---|
| Dataset annotation files | Edit `shell/data/train_datasets.json` and `test_datasets.json` directly with your local `.jsonl` paths |
| InternVL2 pretrained model | Set `MODEL_PATH` env var (default: `pretrained/InternVL2-8B`) |
| Output directory | Set `OUTPUT_DIR` env var (default: `work_dirs/...`) |
| LaBraM pretrained weights | Set `--finetune` in `_build_args()` inside `labram_encoder_fin.py` |
