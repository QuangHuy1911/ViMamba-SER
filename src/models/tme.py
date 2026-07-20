"""
Text-aware Modality Enhancement (TME) module.

Căn chỉnh đặc trưng văn bản theo chiều thời gian của audio bằng
multi-head cross-attention, sau đó dùng cosine-similarity gate
để điều chỉnh mức độ bổ sung thông tin văn bản vào chuỗi audio.

Tham khảo thiết kế: TF-Mamba (EMNLP Findings 2025) — module
text-enhancement với cosine alignment, được điều chỉnh cho
pipeline WavLM + PhoBERT của ViMamba-SER.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextAwareModalityEnhancement(nn.Module):
    """
    Input
    -----
    audio_seq : (B, T_a, D)   — chuỗi embedding audio từ WavLM
    text_seq  : (B, T_t, D)   — chuỗi embedding text từ PhoBERT
    audio_mask: (B, T_a)      — True tại vị trí hợp lệ, False tại padding
    text_mask : (B, T_t)      — True tại vị trí hợp lệ, False tại padding

    Output
    ------
    enhanced  : (B, T_a, D)   — audio đã được bổ sung text
    gate      : (B, T_a, 1)   — giá trị gate ∈ [0, 1] cho visualization
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) phải chia hết cho num_heads ({num_heads})"
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Multi-head cross-attention: audio (query) attends to text (key/value)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm sau cross-attention (pre-norm style)
        self.norm_audio = nn.LayerNorm(embed_dim)
        self.norm_text = nn.LayerNorm(embed_dim)

        # Gate network: cosine similarity → linear → sigmoid
        # Input: cosine similarity (scalar mỗi vị trí) → gate ∈ [0, 1]
        self.gate_linear = nn.Linear(1, 1)

    def forward(self, audio_seq: torch.Tensor, text_seq: torch.Tensor,
                audio_mask: torch.Tensor | None = None,
                text_mask: torch.Tensor | None = None):
        B, T_a, D = audio_seq.shape
        _, T_t, _ = text_seq.shape

        # Normalize trước cross-attention
        audio_normed = self.norm_audio(audio_seq)
        text_normed = self.norm_text(text_seq)

        # nn.MultiheadAttention nhận key_padding_mask: True = ignore
        # Nhưng input mask của ta: True = hợp lệ → đảo ngược
        text_key_padding_mask = None
        if text_mask is not None:
            text_key_padding_mask = ~text_mask  # (B, T_t)

        # Cross-attention: audio query, text key/value
        # Output text_aligned: (B, T_a, D)
        text_aligned, _ = self.cross_attn(
            query=audio_normed,
            key=text_normed,
            value=text_normed,
            key_padding_mask=text_key_padding_mask,
        )

        # Cosine similarity giữa audio và text_aligned theo từng vị trí
        # (B, T_a)
        cos_sim = F.cosine_similarity(audio_seq, text_aligned, dim=-1)

        # Gate: linear + sigmoid → (B, T_a, 1)
        gate = torch.sigmoid(self.gate_linear(cos_sim.unsqueeze(-1)))

        # Mask gate tại vị trí padding audio (gate = 0 tại padding)
        if audio_mask is not None:
            gate = gate * audio_mask.unsqueeze(-1).float()

        # Enhanced = residual connection
        enhanced = audio_seq + gate * text_aligned

        return enhanced, gate
