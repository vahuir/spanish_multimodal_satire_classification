"""
predict_satire.py
-----------------
End-to-end satire detection pipeline for the SatiSPeech dataset.

Given a CSV file with audio paths and transcriptions, this script:
  1. Extracts audio embeddings using the TRILLsson model (Google, TF2).
  2. Extracts text embeddings using a RoBERTa-BNE model (mean pooling).
  3. Runs predictions using one of two supported model backends:
        - "crossattention"  : Cross-attention fusion + feed-forward classifier (PyTorch)
        - "svc"             : Concatenated embeddings + Support Vector Classifier (sklearn)
  4. Saves the predictions to a .pkl file.

No intermediate embedding files are written to disk.

Usage:
    python predict_satire.py \
        --csv         data.csv \
        --audio_dir   audios/ \
        --model_type  crossattention \
        --output      predictions.pkl

    python predict_satire.py \
        --csv         data.csv \
        --audio_dir   audios/ \
        --model_type  svc \
        --output      predictions.pkl
"""

import argparse
import os
import pickle
import numpy as np

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# ── Optional: restrict to a single GPU ────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Dynamic Growth for TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Constants

TEXT_MODEL    = "illuin/roberta-large-bne"
TEXT_DIM      = 1024
TRILLSSON_VER = 3
AUDIO_DIM     = 1024


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 – Audio embedding extraction (TRILLsson)
# ══════════════════════════════════════════════════════════════════════════════

def load_trillsson_model() -> object:
    """
    Download and load the TRILLsson SavedModel from KaggleHub.


    Returns
    -------
    trillsson : TF SavedModel
        Callable TensorFlow model that accepts a batch of waveforms.
    """
    import kagglehub
    model_path = kagglehub.model_download(
        f"google/trillsson/tensorFlow2/{TRILLSSON_VER}"
    )
    return tf.saved_model.load(model_path)


def get_trillsson_embedding(audio_path: str, trillsson_model) -> np.ndarray:
    """
    Extract a TRILLsson embedding from an audio file.

    The audio is resampled to 16 kHz before inference.
    The model returns a (T, 2048) tensor; only the first frame is returned
    as a flat (2048,) vector to keep a fixed-size representation per clip.

    Parameters
    ----------
    audio_path : str
        Path to the audio file (any format supported by librosa).
    trillsson_model :
        Loaded TRILLsson TF SavedModel.

    Returns
    -------
    embedding : np.ndarray of shape (2048,)
        Audio embedding for the given clip.
    """
    waveform, _ = librosa.load(audio_path, sr=16_000)
    waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
    result = trillsson_model(waveform_tensor[tf.newaxis, :])  # add batch dim
    embedding = result["embedding"][0].numpy()  # (T, 2048)
    return embedding  # keep full temporal output; callers may index as needed


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 – Text embedding extraction (RoBERTa-BNE)
# ══════════════════════════════════════════════════════════════════════════════

def load_text_model(model_name: str, device: torch.device):
    """
    Load a HuggingFace tokenizer and encoder model.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g. "illuin/roberta-large-bne".
    device : torch.device
        Device to place the model on.

    Returns
    -------
    tokenizer : PreTrainedTokenizer
    text_model : PreTrainedModel (eval mode, no grad)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_model = AutoModel.from_pretrained(model_name).to(device)
    text_model.eval()
    return tokenizer, text_model


def get_text_embedding(
    text: str,
    tokenizer,
    text_model,
    device: torch.device,
    max_length: int = 128,
) -> torch.Tensor:
    """
    Produce a mean-pooled sentence embedding from a transformer encoder.

    Padding tokens are masked out before averaging so that they do not
    contribute to the final representation.

    Parameters
    ----------
    text : str
        Input transcription.
    tokenizer :
        HuggingFace tokenizer matching the text_model.
    text_model :
        HuggingFace encoder model (must return last_hidden_state).
    device : torch.device
    max_length : int
        Maximum token length (longer sequences are truncated).

    Returns
    -------
    embedding : torch.Tensor of shape (hidden_size,)
        CPU tensor containing the mean-pooled representation.
    """
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        outputs = text_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (1, seq_len, hidden_size)

        # Mask out padding positions before averaging
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, dim=1)
        count = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = summed / count  # (1, hidden_size)

    return mean_pooled.squeeze(0).cpu()  # (hidden_size,)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 – Cross-attention fusion model (PyTorch)
# ══════════════════════════════════════════════════════════════════════════════

class EmbeddingCrossAttention(nn.Module):
    """
    Bidirectional cross-attention between a text embedding and an audio embedding.

    Each modality attends to the other via multi-head attention.  An optional
    gating mechanism linearly interpolates between the original token and its
    context-enriched counterpart.

    Parameters
    ----------
    text_dim : int
        Dimensionality of the input text embedding.
    audio_dim : int
        Dimensionality of the input audio embedding.
    embed_dim : int
        Common projection dimension used by the attention layers.
    num_heads : int
        Number of attention heads.
    use_gate : bool
        If True, apply a sigmoid gate to blend original and attended features.
        If False, apply layer normalisation over the residual sum.
    """

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        embed_dim: int,
        num_heads: int = 4,
        use_gate: bool = True,
    ):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, embed_dim)
        self.audio_proj = nn.Linear(audio_dim, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.use_gate = use_gate

        if use_gate:
            self.gate_t = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid()
            )
            self.gate_a = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid()
            )
        else:
            self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, text_emb: torch.Tensor, audio_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        text_emb  : (batch, text_dim)
        audio_emb : (batch, audio_dim)

        Returns
        -------
        refined_t : (batch, embed_dim)  – text  enriched with audio context
        refined_a : (batch, embed_dim)  – audio enriched with text  context
        """
        t = self.text_proj(text_emb).unsqueeze(1)    # (B, 1, embed_dim)
        a = self.audio_proj(audio_emb).unsqueeze(1)  # (B, 1, embed_dim)

        # Each modality uses the other as key/value
        t_context, _ = self.mha(t, a, a)
        a_context, _ = self.mha(a, t, t)

        if self.use_gate:
            g_t = self.gate_t(torch.cat([t, t_context], dim=-1))
            refined_t = g_t * t + (1 - g_t) * t_context
            g_a = self.gate_a(torch.cat([a, a_context], dim=-1))
            refined_a = g_a * a + (1 - g_a) * a_context
        else:
            refined_t = self.norm(t + t_context)
            refined_a = self.norm(a + a_context)

        return refined_t.squeeze(1), refined_a.squeeze(1)


class MultimodalEmbeddingFusion(nn.Module):
    """
    Fuses text and audio embeddings via cross-attention followed by an MLP.

    Parameters
    ----------
    text_dim   : int  – input text  embedding size
    audio_dim  : int  – input audio embedding size
    embed_dim  : int  – internal projection size for cross-attention
    output_dim : int  – output dimensionality of the fusion MLP
    """

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        embed_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.cross_attn = EmbeddingCrossAttention(text_dim, audio_dim, embed_dim)
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, output_dim),
        )

    def forward(self, text_emb: torch.Tensor, audio_emb: torch.Tensor) -> torch.Tensor:
        ref_t, ref_a = self.cross_attn(text_emb, audio_emb)
        combined = torch.cat([ref_t, ref_a], dim=-1)
        return self.fusion_layer(combined)


class CrossAttentionClassifier(nn.Module):
    """
    Full multimodal classifier: cross-attention fusion → deep feed-forward head.

    Architecture
    ------------
    1. MultimodalEmbeddingFusion  (text + audio → fused vector)
    2. Linear(fusion_dim → 4096) + LayerNorm + GELU + Dropout(0.15)
    3. Linear(4096 → 2048)       + LayerNorm + GELU + Dropout(0.10)
    4. Linear(2048 → 512)
    5. Linear(512 → num_classes)

    Parameters
    ----------
    audio_dim   : int  – audio embedding dimensionality  (TRILLsson: 1024)
    text_dim    : int  – text  embedding dimensionality  (RoBERTa-L: 1024)
    fusion_dim  : int  – shared projection / fusion size (default: 512)
    num_classes : int  – number of output classes        (default: 2)
    """

    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        fusion_dim: int = 512,
        num_classes: int = 2,
    ):
        super().__init__()
        self.fusion = MultimodalEmbeddingFusion(
            text_dim=text_dim,
            audio_dim=audio_dim,
            embed_dim=fusion_dim,
            output_dim=fusion_dim,
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 4096),
            nn.LayerNorm(4096),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(4096, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(2048, 512),
            nn.Linear(512, num_classes),
        )

    def forward(self, audio_x: torch.Tensor, text_x: torch.Tensor) -> torch.Tensor:
        fused = self.fusion(text_x, audio_x)
        return self.classifier(fused)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 – Prediction helpers
# ══════════════════════════════════════════════════════════════════════════════

def predict_crossattention(
    audio_embedding: np.ndarray,
    transcription: str,
    model: CrossAttentionClassifier,
    tokenizer,
    text_model,
    device: torch.device,
) -> tuple[str, float]:
    """
    Predict satire/no-satire for a single sample using the cross-attention model.

    Parameters
    ----------
    audio_embedding : np.ndarray
        Pre-extracted TRILLsson embedding for this sample.
    transcription : str
        Text transcription of the audio.
    model : CrossAttentionClassifier
        Loaded and eval-mode PyTorch model.
    tokenizer, text_model :
        HuggingFace tokenizer and encoder.
    device : torch.device

    Returns
    -------
    label      : str    – "satire" or "no-satire"
    confidence : float  – softmax probability of the predicted class
    """
    model.eval()

    # Flatten audio embedding if it has extra dimensions
    emb_audio = torch.from_numpy(audio_embedding).float()
    if emb_audio.ndim > 1:
        emb_audio = emb_audio.view(-1)

    emb_text = get_text_embedding(transcription, tokenizer, text_model, device)

    emb_audio = emb_audio.unsqueeze(0).to(device)
    emb_text = emb_text.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(emb_audio, emb_text)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()

    label = "satire" if pred == 1 else "no-satire"
    return label, confidence


def predict_svc(
    audio_embedding: np.ndarray,
    transcription: str,
    classifiers: list,
    scaler,
    tokenizer,
    text_model,
    device: torch.device,
    voting: bool = False
) -> str:
    """
    Predict satire/no-satire for a single sample using the SVC pipeline.

    The audio and text embeddings are concatenated, scaled, and passed to the
    first SVC model in the list.

    Parameters
    ----------
    audio_embedding : np.ndarray
        Pre-extracted TRILLsson embedding for this sample.
    transcription : str
        Text transcription of the audio.
    classifiers : list
        List of fitted sklearn SVC models.
    scaler :
        Fitted sklearn scaler (e.g. StandardScaler).
    tokenizer, text_model :
        HuggingFace tokenizer and encoder.
    device : torch.device
    voting: bool
        Enables voting

    Returns
    -------
    label : str – "satire" or "no-satire"
    """
    emb_audio = torch.from_numpy(audio_embedding).float()
    emb_text = get_text_embedding(transcription, tokenizer, text_model, device)

    combined = torch.cat([emb_audio, emb_text], dim=-1).numpy()
    combined = scaler.transform(combined.reshape(1, -1))

    preds = []

    for classifier in classifiers:
        pred = classifier.predict(combined)[0]
        preds.append(pred)

        if not voting:
            break

    return "satire" if np.median(preds) == 1 else "no-satire"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 – CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end satire detection pipeline."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the CSV file.",
    )
    parser.add_argument(
        "--id_key",
        help="Column that contains the ID of the sample.",
        default="uid"
    )
    parser.add_argument(
        "--text_key",
        help="Column that contains the text of the sample.",
        default="transcription"
    )
    parser.add_argument(
        "--audio_key",
        help="Column that contains the path to the audio file.",
    )
    parser.add_argument(
        "--audio_dir",
        default="audios/",
        help="Directory containing the audio files referenced by 'uid'.",
    )
    parser.add_argument(
        "--model_type",
        choices=["crossattention", "svc"],
        default="svc",
        help="Which classifier backend to use.",
    )
    parser.add_argument(
        "--crossattn_model",
        default="models/RoBERTa-BNE_TRILLsson_CrossAttention/model.pth",
        help="Path to the cross-attention model weights (.pth). "
             "Used only when --model_type=crossattention.",
    )
    parser.add_argument(
        "--svc_models",
        default="models/RoBERTa-BNE_TRILLsson_SVC/models.pkl",
        help="Path to the SVC models pickle. "
             "Used only when --model_type=svc.",
    )
    parser.add_argument(
        "--svc_scaler",
        default="models/RoBERTa-BNE_TRILLsson_SVC/scaler.pkl",
        help="Path to the feature scaler pickle. "
             "Used only when --model_type=svc.",
    )
    parser.add_argument(
        "--svc_voting",
        action="store_true",
        help="Uses voting for prediction in SVC",
        default=False
    )
    parser.add_argument(
        "--output",
        default="predictions.pkl",
        help="Path for the output predictions pickle file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"Loading dataset from: {args.csv}")
    df = pd.read_csv(args.csv)

    # ── Load TRILLsson ────────────────────────────────────────────────────────
    print("Loading TRILLsson model…")
    trillsson = load_trillsson_model()

    # ── Load text model ───────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading text model: {TEXT_MODEL}")
    tokenizer, text_model = load_text_model(TEXT_MODEL, device)

    ca_classifier, svc_classifiers, scaler = None, None, None

    # ── Load classifier ───────────────────────────────────────────────────────
    if args.model_type == "crossattention":
        print(f"Loading cross-attention model from: {args.crossattn_model}")
        ca_classifier = CrossAttentionClassifier(
            audio_dim=AUDIO_DIM,
            text_dim=TEXT_DIM,
            fusion_dim=512,
        )
        ca_classifier.load_state_dict(
            torch.load(args.crossattn_model, map_location=device)
        )
        ca_classifier.to(device)
        ca_classifier.eval()

    else:
        print(f"Loading SVC models from: {args.svc_models}")
        with open(args.svc_models, "rb") as f:
            svc_classifiers = pickle.load(f)

        print(f"Loading scaler from: {args.svc_scaler}")
        with open(args.svc_scaler, "rb") as f:
            scaler = pickle.load(f)

    # ── Run inference ─────────────────────────────────────────────────────────
    predictions = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        if args.audio_key is None:
            audio_path = os.path.join(args.audio_dir, row[args.id_key])

        else:
            audio_path = os.path.join(
                args.audio_dir, row[args.audio_key]
            )

        # 1. Extract audio embedding (no disk save)
        audio_emb = get_trillsson_embedding(audio_path, trillsson)

        # 2. Predict
        if args.model_type == "crossattention":
            label, _ = predict_crossattention(
                audio_embedding=audio_emb,
                transcription=row[args.text_key],
                model=ca_classifier,
                tokenizer=tokenizer,
                text_model=text_model,
                device=device,
            )
        else:
            assert isinstance(svc_classifiers, list)

            label = predict_svc(
                audio_embedding=audio_emb,
                transcription=row[args.text_key],
                classifiers=svc_classifiers,
                scaler=scaler,
                tokenizer=tokenizer,
                text_model=text_model,
                device=device,
                voting=args.svc_voting
            )

        predictions.append(label)

    # ── Save predictions ──────────────────────────────────────────────────────
    print(f"Saving predictions to: {args.output}")
    with open(args.output, "wb") as f:
        pickle.dump(predictions, f)

    satire_count = predictions.count("satire")
    print(
        f"Done. {len(predictions)} predictions saved "
        f"({satire_count} satire, {len(predictions) - satire_count} no-satire)."
    )


if __name__ == "__main__":
    main()
