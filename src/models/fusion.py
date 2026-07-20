"""
Bi-directional Sequence Fusion module.

Xử lý chuỗi enhanced embedding theo hai chiều thời gian, rồi pool
thành vector cố định. Ưu tiên dùng Bi-Mamba (state-space model);
nếu mamba-ssm không có sẵn (thường xảy ra trên CPU hoặc Colab
không cài được), tự động fallback sang BiGRU 2 lớp.

Có flag `force_fallback` để ép dùng BiGRU ngay cả khi mamba-ssm
đã cài — phục vụ ablation so sánh kiến trúc.
"""

import warnings
import torch
import torch.nn as nn

# Thử import mamba_ssm ở module-level, ghi nhận trạng thái
_MAMBA_AVAILABLE = False
try:
    from mamba_ssm import Mamba
    _MAMBA_AVAILABLE = True
except ImportError:
    _MAMBA_AVAILABLE = False


def is_mamba_available() -> bool:
    """Kiểm tra mamba-ssm đã cài chưa (utility cho code bên ngoài)."""
    return _MAMBA_AVAILABLE


class BiMambaBlock(nn.Module):
    """
    Bi-directional Mamba: chạy 2 khối Mamba riêng biệt trên chuỗi
    xuôi và chuỗi đảo ngược, rồi cộng (sum) 2 output.

    Chỉ khởi tạo được khi mamba-ssm đã import thành công.
    """

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        if not _MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm chưa được cài đặt. Gọi BiSequenceFusion "
                "với force_fallback=True hoặc cài mamba-ssm."
            )
        self.mamba_fwd = Mamba(
            d_model=d_model, d_state=d_state,
            d_conv=d_conv, expand=expand,
        )
        self.mamba_bwd = Mamba(
            d_model=d_model, d_state=d_state,
            d_conv=d_conv, expand=expand,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x    : (B, T, D)
        mask : (B, T) — True tại vị trí hợp lệ (chưa dùng trực tiếp
               trong Mamba vì SSM xử lý tuần tự, nhưng giữ interface
               nhất quán; zero-out padding ở bước pooling)
        """
        # Forward pass
        out_fwd = self.mamba_fwd(x)  # (B, T, D)

        # Backward pass: đảo ngược chuỗi theo chiều thời gian
        x_rev = torch.flip(x, dims=[1])
        out_bwd = torch.flip(self.mamba_bwd(x_rev), dims=[1])

        # Sum 2 chiều + residual + norm
        out = self.norm(x + out_fwd + out_bwd)
        return out


class BiGRUBlock(nn.Module):
    """
    Fallback: BiGRU 2 lớp, hidden size sao cho output dimension
    khớp với d_model (mỗi chiều = d_model // 2).
    """

    def __init__(self, d_model: int, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % 2 == 0, (
            f"d_model ({d_model}) phải chẵn để chia đều cho 2 chiều GRU"
        )
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x    : (B, T, D)
        mask : (B, T) — True tại vị trí hợp lệ
        """
        # Pack padded sequences nếu có mask, để GRU bỏ qua padding
        if mask is not None:
            lengths = mask.sum(dim=1).to("cpu", dtype=torch.int64)  # (B,)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.clamp(min=1), batch_first=True,
                enforce_sorted=False,
            )
            out_packed, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out_packed, batch_first=True, total_length=x.size(1)
            )
        else:
            out, _ = self.gru(x)

        # Residual + norm
        out = self.norm(x + out)
        return out


class BiSequenceFusion(nn.Module):
    """
    Nhận chuỗi enhanced (B, T, D), xử lý bi-directional, rồi pool
    thành vector (B, output_dim).

    Output dim = 2 * hidden_dim (concat mean-pool và max-pool).

    Parameters
    ----------
    hidden_dim     : int  — kích thước ẩn (= d_model cho Mamba, hoặc
                     input/output dim cho BiGRU)
    force_fallback : bool — ép dùng BiGRU kể cả khi mamba-ssm có sẵn
    mamba_d_state  : int  — state dimension cho Mamba (mặc định 16)
    mamba_d_conv   : int  — conv kernel size cho Mamba (mặc định 4)
    mamba_expand   : int  — expansion factor cho Mamba (mặc định 2)
    gru_num_layers : int  — số lớp GRU fallback (mặc định 2)
    dropout        : float — dropout cho GRU fallback
    """

    def __init__(self, hidden_dim: int = 768,
                 force_fallback: bool = False,
                 mamba_d_state: int = 16,
                 mamba_d_conv: int = 4,
                 mamba_expand: int = 2,
                 gru_num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.using_mamba = False

        if _MAMBA_AVAILABLE and not force_fallback:
            self.seq_block = BiMambaBlock(
                d_model=hidden_dim,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
            )
            self.using_mamba = True
        else:
            if not _MAMBA_AVAILABLE and not force_fallback:
                warnings.warn(
                    "mamba-ssm không có sẵn, dùng fallback BiGRU — "
                    "cài mamba-ssm trên Colab GPU để bật kiến trúc đầy đủ",
                    UserWarning,
                    stacklevel=2,
                )
            self.seq_block = BiGRUBlock(
                d_model=hidden_dim,
                num_layers=gru_num_layers,
                dropout=dropout,
            )

        # Output dimension: concat mean-pool + max-pool
        self.output_dim = hidden_dim * 2

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x    : (B, T, D) — chuỗi enhanced từ TME
        mask : (B, T)    — True tại vị trí hợp lệ

        Returns
        -------
        pooled : (B, 2*D) — concat của mean-pool và max-pool
        """
        # Bi-directional sequence processing
        seq_out = self.seq_block(x, mask=mask)  # (B, T, D)

        # Masked pooling
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()  # (B, T, 1)
            # Mean pool: chỉ tính trên vị trí hợp lệ
            lengths = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
            mean_pool = (seq_out * mask_expanded).sum(dim=1) / lengths.squeeze(-1)

            # Max pool: đặt padding = -inf trước khi lấy max
            seq_masked = seq_out.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            max_pool, _ = seq_masked.max(dim=1)
        else:
            mean_pool = seq_out.mean(dim=1)  # (B, D)
            max_pool, _ = seq_out.max(dim=1)  # (B, D)

        # Concat 2 loại pooling → (B, 2*D)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        return pooled
