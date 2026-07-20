"""
ViMamba-SER: pipeline end-to-end cho nhận dạng cảm xúc giọng nói
tiếng Việt.

Ghép nối:
  WavLM-Base (frozen) → sequence audio embedding
  PhoWhisper → PhoBERT-v2 (frozen) → sequence text embedding
  TME (cross-attention alignment + cosine gate)
  BiSequenceFusion (Bi-Mamba hoặc BiGRU fallback)
  MLP classifier (3 lớp, dropout 0.3, 4 classes)

Module này có 2 chế độ:
  - Full pipeline: nhận raw waveform + transcript, chạy encoder →
    TME → Fusion → classify. Dùng khi inference (demo Gradio).
  - Embedding pipeline: nhận pre-extracted sequence embeddings,
    chạy TME → Fusion → classify. Dùng khi train (nhanh hơn vì
    không cần chạy lại encoder mỗi epoch).
"""

import torch
import torch.nn as nn

from src.models.tme import TextAwareModalityEnhancement
from src.models.fusion import BiSequenceFusion


class ViMambaSERClassifier(nn.Module):
    """
    Phần trainable: TME + Fusion + MLP classifier.

    Nhận embeddings đã trích sẵn (không bao gồm WavLM/PhoBERT,
    vì các model này frozen và chạy riêng ở bước feature extraction).

    Parameters
    ----------
    embed_dim      : int  — chiều embedding (768 cho WavLM/PhoBERT)
    num_classes    : int  — số lớp cảm xúc (4 cho ViSEC)
    tme_num_heads  : int  — số head cross-attention trong TME
    tme_dropout    : float — dropout cho cross-attention
    fusion_hidden  : int  — hidden dim cho fusion module
    fusion_force_fallback : bool — ép dùng BiGRU fallback
    mlp_dropout    : float — dropout cho MLP classifier
    mamba_d_state  : int  — state dim cho Mamba (nếu dùng)
    mamba_d_conv   : int  — conv kernel size cho Mamba (nếu dùng)
    mamba_expand   : int  — expansion factor cho Mamba (nếu dùng)
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 4,
        tme_num_heads: int = 8,
        tme_dropout: float = 0.1,
        fusion_hidden: int = 768,
        fusion_force_fallback: bool = False,
        mlp_dropout: float = 0.3,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # TME module
        self.tme = TextAwareModalityEnhancement(
            embed_dim=embed_dim,
            num_heads=tme_num_heads,
            dropout=tme_dropout,
        )

        # Projection nếu fusion_hidden != embed_dim
        self.proj = None
        if fusion_hidden != embed_dim:
            self.proj = nn.Linear(embed_dim, fusion_hidden)

        # Fusion module
        self.fusion = BiSequenceFusion(
            hidden_dim=fusion_hidden,
            force_fallback=fusion_force_fallback,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
        )

        # MLP classifier: 3 lớp, giữ đúng kiến trúc Phase A midterm
        # Linear → ReLU → Dropout → Linear → ReLU → Linear → num_classes
        fusion_output_dim = self.fusion.output_dim  # 2 * fusion_hidden
        mlp_hidden = fusion_output_dim  # lớp ẩn đầu tiên giữ nguyên dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_output_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden // 2, num_classes),
        )

    def forward(
        self,
        audio_seq: torch.Tensor,
        text_seq: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        audio_seq  : (B, T_a, D) — sequence embeddings từ WavLM
        text_seq   : (B, T_t, D) — sequence embeddings từ PhoBERT
        audio_mask : (B, T_a) — True tại vị trí hợp lệ
        text_mask  : (B, T_t) — True tại vị trí hợp lệ

        Returns
        -------
        dict chứa:
          "logits"   : (B, num_classes) — logits chưa softmax
          "gate"     : (B, T_a, 1) — gate values từ TME (cho visualization)
        """
        # TME: audio + text → enhanced audio
        enhanced, gate = self.tme(
            audio_seq, text_seq,
            audio_mask=audio_mask,
            text_mask=text_mask,
        )

        # Projection nếu cần
        if self.proj is not None:
            enhanced = self.proj(enhanced)

        # Fusion: sequence → fixed-size vector
        # Dùng audio_mask vì enhanced có cùng chiều T_a với audio
        pooled = self.fusion(enhanced, mask=audio_mask)

        # Classify
        logits = self.classifier(pooled)

        return {"logits": logits, "gate": gate}
