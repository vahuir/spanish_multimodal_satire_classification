# atire Detection Pipeline

End-to-end inference script for multimodal (audio + text) satire detection in Spanish fine-tuned on the **SatiSPeech** dataset. Given a CSV file, it extracts audio and text embeddings on-the-fly and produces per-sample predictions without writing any intermediate files to disk.

---

## Overview

The pipeline supports two classifier backends that share the same embedding extraction front-end:

| Backend | Audio model | Text model | Fusion |
|---|---|---|---|
| `crossattention` | TRILLsson (TF2) | RoBERTa-large-BNE | Cross-attention + feed-forward head |
| `svc` | TRILLsson (TF2) | RoBERTa-large-BNE | Concatenation + sklearn SVC |

---

## Repository structure

```
.
├── predict_satire.py          # main script (this repo)
└── models/
    ├── multimodal_crossattention.pth       # cross-attention weights
    ├── multimodal_svc.pkl                  # multimodal SVC
    └── audio_scaler.pkl                    # feature scaler for SVC
```

---

## Requirements

```
python >= 3.10
torch
transformers
tensorflow
librosa
kagglehub
scikit-learn
pandas
tqdm
```

Install all dependencies at once:

```bash
pip install torch transformers tensorflow librosa kagglehub scikit-learn pandas tqdm
```

> **Note:** TRILLsson is downloaded automatically from KaggleHub on first run. Make sure you have valid Kaggle credentials configured (`~/.kaggle/kaggle.json`).

---

## Usage

### Cross-attention classifier (default)

```bash
python predict_satire.py \
    --csv         SatiSPeech_phase_2_test_public.csv \
    --audio_dir   audios/ \
    --model_type  crossattention \
    --crossattn_weights modelos/trillssonBNE_crossattention_weights.pth \
    --output      predictions_crossattn.pkl
```

### SVC classifier

```bash
python predict_satire.py \
    --csv         SatiSPeech_phase_2_test_public.csv \
    --audio_dir   audios/ \
    --model_type  svc \
    --svc_models  modelos/modelos_TRILLssonBNE.pkl \
    --svc_scaler  modelos/scalerTRILLssonBNE.pkl \
    --output      predictions_svc.pkl
```

---

## All CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--csv` | *(required)* | Path to the input CSV. Must have `uid` and `transcription` columns. |
| `--audio_dir` | `audios/` | Directory containing the audio files referenced by `uid`. |
| `--model_type` | `crossattention` | Classifier backend: `crossattention` or `svc`. |
| `--crossattn_model` | `models/RoBERTa-BNE_TRILLsson_CrossAttention/model.pth` | Path to cross-attention `.pth` weights. |
| `--svc_models` | `models/RoBERTa-BNE_TRILLsson_SVC/models.pk` | Path to the SVC models pickle. |
| `--svc_scaler` | `models/RoBERTa-BNE_TRILLsson_SVC/scaler.pkl` | Path to the feature 
| `--svc_voting` | `False` | If enabled, all the SVC models are used 
| `--output` | `predictions.pkl` | Output path for the predictions pickle. |

---

## Input CSV format

The CSV must contain at least these two columns:

| Column | Description |
|---|---|
| `uid` | Filename of the audio file (e.g. `sample_001.mp3`). Joined with `--audio_dir`. |
| `transcription` | Text transcription of the audio clip. |

---

## Output format

The output is a Python list serialised with `pickle`, where each element is either `"satire"` or `"no-satire"`, in the same order as the input CSV rows.

```python
import pickle

with open("predictions.pkl", "rb") as f:
    predictions = pickle.load(f)

# ['no-satire', 'satire', 'no-satire', ...]
```

---

## Architecture notes

### TRILLsson (audio)

TRILLsson is a speech representation model from Google that produces a sequence of frame-level embeddings of shape `(T, 1024)`. The full temporal output is passed to the classifier, which flattens it when needed.

### RoBERTa-large-BNE (text)

[`illuin/roberta-large-bne`](https://huggingface.co/illuin/roberta-large-bne) is a Spanish RoBERTa model trained on the BNE corpus. Text embeddings are computed via **mean pooling** over the last hidden states of all non-padding tokens, yielding a single `(1024,)` vector per sample.

### Cross-attention fusion model

```
text_emb  ──┐
            ├─► EmbeddingCrossAttention ─► concat ─► MLP ─► fused_vec
audio_emb ──┘                                                  │
                                                               ▼
                                              deep FFN (4096→2048→512→2) ─► logits
```

Each modality attends to the other via `nn.MultiheadAttention`. A sigmoid gate interpolates between the original embedding and its attention-enriched counterpart, letting the model learn how much cross-modal context to incorporate.

### SVC pipeline

Audio and text embeddings are concatenated into a single feature vector `(1024 + 1024 = 2048,)`, standardised with the saved `StandardScaler`, and classified by a pre-trained `SVC`.

---

## GPU / CPU

The script automatically uses CUDA if available, falling back to CPU otherwise. TRILLsson runs on TensorFlow (GPU 0 is selected via `CUDA_VISIBLE_DEVICES`); the PyTorch models respect the same device.


## Citation

If you use this work or the associated code in your research, please cite it as follows:

```bibtex
@unpublished{ahuir2026listen,
  author    = {Ahuir, Vicent and Barcel{\'o}-Milkova, Alejandro Joaqu{\'\i}n and Casamayor-Segarra, Andreu and Hurtado, Llu{\'\i}s-F. and Castro-Bleda, Maria Jose},
  title     = {Listen, Read, and Detect: Multimodal Satire Classification in Spanish Language Media},
  note      = {Valencia Research Institute for Artificial Intelligence (VRAIN), Universitat Polit{\`e}cnica de Val{\`e}ncia},
  address   = {Val{\`e}ncia, Spain},
  year      = {2026}
}
