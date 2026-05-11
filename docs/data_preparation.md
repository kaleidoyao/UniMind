# Data Preparation Guide

This guide explains how to prepare EEG data for UniMind training and evaluation.

## Overview

The full pipeline has three steps:

```
Raw EEG files  →  [Step 1] Preprocess & segment       →  .pkl files
                  [Step 2] Generate JSONL annotation  →  .jsonl files
                  [Step 3] Place in instruction_jsonl/
```

---

## Step 1: Preprocess Raw EEG → `.pkl` Files

Each `.pkl` file stores one EEG segment and must contain:

```python
{"X": np.ndarray}   # shape: [C, T], dtype: float32
                    # C = number of channels, T = number of time points
```

### 1.1 For `.edf` / `.cnt` / `.gdf` files (mne-supported formats)

```python
import mne
import pickle
import numpy as np

def preprocess_edf(file_path, l_freq=0.1, h_freq=75.0, target_sfreq=200,
                   keep_channels=None, drop_channels=None):
    raw = mne.io.read_raw_edf(file_path, preload=True)

    # Drop unwanted channels
    if drop_channels:
        to_drop = [ch for ch in drop_channels if ch in raw.ch_names]
        raw.drop_channels(to_drop)

    # Keep only target channels (optional)
    if keep_channels:
        raw.pick_channels(keep_channels)

    # Bandpass filter: 0.1–75 Hz
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    # Notch filter: 50 Hz powerline
    raw.notch_filter(50.0)
    # Resample to 200 Hz
    raw.resample(target_sfreq)

    # Get data in µV, shape: [C, T]
    data = raw.get_data(units='uV').astype(np.float32)
    return data, raw.ch_names
```

> **Note**: If the raw sampling rate is low (e.g., 100 Hz), resample to 200 Hz **before** filtering to avoid Nyquist errors.

### 1.2 For `.mat` files (e.g., SEED, SEED-IV)

```python
import scipy.io as sio
import mne
import numpy as np

def preprocess_mat(file_path, orig_sfreq=200, l_freq=0.1, h_freq=75.0):
    data = sio.loadmat(file_path)
    # SEED: keys like "eeg1", "eeg2", ..., "eeg15" (one per trial)
    trial_keys = [k for k in data.keys() if "eeg" in k.lower()]

    segments = []
    for key in trial_keys:
        raw = data[key].astype(np.float64)          # [C, T]
        raw = mne.filter.filter_data(raw, orig_sfreq, l_freq, h_freq)
        raw = mne.filter.notch_filter(raw, Fs=orig_sfreq, freqs=50.0)
        # Only resample if orig_sfreq != 200
        # raw = mne.filter.resample(raw, up=200, down=orig_sfreq)
        segments.append(raw.astype(np.float32))

    return segments
```

### 1.3 Segment and save as `.pkl`

For dataset-specific segmentation details (window sizes, stride, trial boundaries), refer to the preprocessing scripts in the [LaBraM repository](https://github.com/935963004/LaBraM).

---

## Step 2: Generate JSONL Annotation Files

### JSONL Format

Each line is one JSON object:

```json
{
  "id": "000000",
  "label": 0,
  "EEG": "path/to/sample.pkl",
  "conversations": [
    {"from": "human", "value": "<EEG>\n{question}"},
    {"from": "gpt",   "value": "{answer}"}
  ]
}
```

**Field descriptions:**

| Field | Type | Description |
|---|---|---|
| `id` | string | Zero-padded index, e.g. `"000042"` |
| `label` | int | Integer class index (used for evaluation metrics) |
| `EEG` | string | Path to `.pkl` file (relative to working directory, or absolute) |
| `conversations[0].value` | string | Must start with `<EEG>\n`, followed by the instruction prompt |
| `conversations[1].value` | string | Expected model output (class name string) |

### Generation script

```python
import os
import json
import pickle

def make_jsonl(pkl_dir, save_path, label_map, question_template):
    """
    pkl_dir          : directory containing .pkl files
    save_path        : output .jsonl file path
    label_map        : dict mapping int label → answer string, e.g. {0: "positive", 1: "negative"}
    question_template: instruction string, e.g. "What emotion does this EEG signal represent? [positive, negative, neutral]"
    """
    pkl_files = sorted([f for f in os.listdir(pkl_dir) if f.endswith('.pkl')])

    with open(save_path, 'w') as out:
        for idx, fname in enumerate(pkl_files):
            file_path = os.path.join(pkl_dir, fname)
            sample = pickle.load(open(file_path, 'rb'))
            label = int(sample['Y'])

            entry = {
                "id": f"{idx:06d}",
                "label": label,
                "EEG": file_path,
                "conversations": [
                    {"from": "human", "value": f"<EEG>\n{question_template}"},
                    {"from": "gpt",   "value": label_map[label]},
                ]
            }
            out.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Saved {len(pkl_files)} entries to {save_path}")
```

### Example: SEED dataset

```python
make_jsonl(
    pkl_dir="data/SEED/train",
    save_path="instruction_jsonl/SEED_train.jsonl",
    label_map={0: "positive", 1: "negative", 2: "neutral"},
    question_template="From this EEG signal, identify the emotion. [positive, negative, neutral]",
)
```

---

## Step 3: Dataset Config Template

After generating JSONL files, configure the dataset JSON used by the training/evaluation scripts.

### Training config template (`shell/data/train_datasets.json`)

```json
{
  "DATASET_NAME": {
    "root": ".",
    "annotation": "your/path/to/instruction_jsonl/DATASET_train.jsonl",
    "task_caption": "task description",
    "data_augment": false,
    "repeat_time": 1,
    "sample_rate": 200,
    "num_image_token": 63,
    "ch_names": ["CH1", "CH2", "..."],
    "mean": [0.0, 0.0, "..."],
    "std":  [1.0, 1.0, "..."]
  }
}
```

### Evaluation config template (`shell/data/test_datasets.json`)

```json
{
  "DATASET_NAME": {
    "train": "your/path/to/instruction_jsonl/DATASET_train.jsonl",
    "test":  "your/path/to/instruction_jsonl/DATASET_test.jsonl",
    "task_caption": "task description",
    "metric": "accuracy",
    "max_new_tokens": 10,
    "sample_rate": 200,
    "num_image_token": 63,
    "ch_names": ["CH1", "CH2", "..."],
    "mean": [0.0, 0.0, "..."],
    "std":  [1.0, 1.0, "..."]
  }
}
```

### Computing `mean` and `std`

```python
import pickle, numpy as np, glob

files = glob.glob("instruction_jsonl/SEED_train.jsonl")
all_data = []
import json
for line in open(files[0]):
    entry = json.loads(line)
    x = pickle.load(open(entry['EEG'], 'rb'))
    data = x['X'] if isinstance(x, dict) else x   # [C, T]
    all_data.append(data)

all_data = np.concatenate(all_data, axis=1)   # [C, total_T]
mean = all_data.mean(axis=1).tolist()
std  = all_data.std(axis=1).tolist()
print("mean:", mean)
print("std:", std)
```
