"""Neural network architectures for Songsplat.

Mode 1 - PretrainedBackbone:
    Uses a HuggingFace wav2vec2/HuBERT encoder (frozen or fine-tunable) +
    lightweight MLP regression heads per splat.

Mode 2 - RawTransformer:
    Log-mel spectrogram → patch embeddings → transformer encoder →
    per-splat regression heads. Trained from scratch.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared regression head
# ---------------------------------------------------------------------------

class SplatHead(nn.Module):
    """Single MLP regression head → scalar in [0, 1]."""

    def __init__(self, in_features: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Mode 1 - Pretrained backbone
# ---------------------------------------------------------------------------

class PretrainedBackbone(nn.Module):
    """Wav2Vec2/HuBERT feature extractor + per-splat MLP heads.

    The backbone is frozen by default; set fine_tune=True to unfreeze it.
    Input audio is automatically resampled from SOURCE_SR to 16 kHz as
    required by Wav2Vec2.
    """

    BACKBONE_MODEL = "facebook/wav2vec2-base"
    TARGET_SR = 16000  # Wav2Vec2 requires 16kHz

    def __init__(
        self,
        num_splats: int,
        fine_tune: bool = False,
        model_name: Optional[str] = None,
        source_sr: int = 22050,
    ) -> None:
        super().__init__()
        self._num_splats = num_splats
        model_name = model_name or self.BACKBONE_MODEL

        try:
            from transformers import Wav2Vec2Model  # type: ignore
            # ignore_mismatched_sizes silences the quantizer-key warnings from
            # checkpoints that were saved with Wav2Vec2ForPreTraining weights.
            self.backbone = Wav2Vec2Model.from_pretrained(
                model_name, ignore_mismatched_sizes=True
            )
            self._embed_dim = self.backbone.config.hidden_size
        except ImportError:
            raise RuntimeError(
                "transformers is required for Mode 1. "
                "Install with: pip install transformers"
            )

        if not fine_tune:
            for param in self.backbone.parameters():
                param.requires_grad = False
        if not fine_tune:
            self.backbone.eval()

        self.heads = nn.ModuleList([
            SplatHead(self._embed_dim) for _ in range(num_splats)
        ])

        # Registered resample transform so it moves with .to(device)
        try:
            import torchaudio  # type: ignore
            if source_sr != self.TARGET_SR:
                self.resample = torchaudio.transforms.Resample(
                    orig_freq=source_sr, new_freq=self.TARGET_SR
                )
            else:
                self.resample = None
        except ImportError:
            self.resample = None

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        audio: (B, T) float32 waveform at source_sr
        returns: (B, num_splats) predictions in [0, 1]
        """
        if self.resample is not None:
            audio = self.resample(audio)
        outputs = self.backbone(audio)
        # Mean-pool over time dimension
        features = outputs.last_hidden_state.mean(dim=1)  # (B, embed_dim)
        preds = torch.stack([head(features) for head in self.heads], dim=1)  # (B, S)
        return preds


# ---------------------------------------------------------------------------
# Mode 2 - Raw transformer
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """Convert log-mel spectrogram to patch tokens."""

    def __init__(
        self,
        n_mels: int = 80,
        patch_size: int = 16,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=(n_mels, patch_size), stride=(n_mels, patch_size))

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """spec: (B, 1, n_mels, T_frames) → (B, n_patches, embed_dim)"""
        x = self.proj(spec)          # (B, embed_dim, 1, n_patches)
        x = x.squeeze(2)             # (B, embed_dim, n_patches)
        x = x.permute(0, 2, 1)       # (B, n_patches, embed_dim)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 2048) -> None:
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class RawTransformer(nn.Module):
    """Spectrogram-patch transformer for multi-splat regression."""

    def __init__(
        self,
        num_splats: int,
        sample_rate: int = 22050,
        n_mels: int = 80,
        patch_size: int = 16,
        embed_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._sample_rate = sample_rate
        self._n_mels = n_mels

        try:
            import torchaudio  # type: ignore
            # Register as submodules so .to(device) propagates to them
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=1024,
                hop_length=160,
                n_mels=n_mels,
                power=2.0,
            )
            self.db_transform = torchaudio.transforms.AmplitudeToDB()
        except ImportError:
            raise RuntimeError(
                "torchaudio is required for Mode 2. "
                "Install with: pip install torchaudio"
            )

        self.patch_embed = PatchEmbedding(n_mels=n_mels, patch_size=patch_size, embed_dim=embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

        self.heads = nn.ModuleList([
            SplatHead(embed_dim) for _ in range(num_splats)
        ])

    def _to_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (B, T) → (B, 1, n_mels, T_frames)"""
        spec = self.mel_transform(audio)         # (B, n_mels, T_frames)
        spec = self.db_transform(spec)           # log scale
        spec = spec.unsqueeze(1)                 # (B, 1, n_mels, T_frames)
        return spec

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (B, T) → (B, num_splats) predictions in [0, 1]"""
        spec = self._to_spectrogram(audio)       # (B, 1, n_mels, T_frames)
        patches = self.patch_embed(spec)         # (B, n_patches, embed_dim)

        # Prepend CLS token
        B = patches.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1)     # (B, 1 + n_patches, embed_dim)
        x = self.pos_enc(x)

        x = self.transformer(x)
        x = self.norm(x)
        cls_out = x[:, 0]                        # (B, embed_dim)

        preds = torch.stack([head(cls_out) for head in self.heads], dim=1)  # (B, S)
        return preds


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(
    architecture: str,
    num_splats: int,
    **kwargs,
) -> nn.Module:
    if architecture == "pretrained":
        return PretrainedBackbone(num_splats=num_splats, **kwargs)
    elif architecture == "raw_transformer":
        return RawTransformer(num_splats=num_splats, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
