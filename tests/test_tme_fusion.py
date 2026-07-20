"""
Unit tests cho TME + Fusion + ViMambaSER pipeline.

Chạy hoàn toàn trên CPU với tensor ngẫu nhiên, không cần model thật
hay GPU. Kiểm tra:
  - Shape đầu ra đúng
  - Gate nằm trong [0, 1]
  - Forward pass không lỗi cả khi mamba_ssm không có (mock fallback)
  - Gradient chảy ngược được (loss.backward())
  - Masked inputs xử lý đúng
"""

import sys
import unittest
from unittest import mock

import torch
import torch.nn as nn

# Thêm project root vào path
sys.path.insert(0, ".")

from src.models.tme import TextAwareModalityEnhancement
from src.models.fusion import BiSequenceFusion, BiGRUBlock, _MAMBA_AVAILABLE
from src.models.vimamba_ser import ViMambaSERClassifier


class TestTME(unittest.TestCase):
    """Tests cho TextAwareModalityEnhancement."""

    def setUp(self):
        self.B = 4
        self.T_a = 50
        self.T_t = 12
        self.D = 768
        self.tme = TextAwareModalityEnhancement(
            embed_dim=self.D, num_heads=8, dropout=0.1
        )
        self.tme.eval()

    def test_output_shape(self):
        """Enhanced output và gate có shape đúng."""
        audio = torch.randn(self.B, self.T_a, self.D)
        text = torch.randn(self.B, self.T_t, self.D)

        enhanced, gate = self.tme(audio, text)

        self.assertEqual(enhanced.shape, (self.B, self.T_a, self.D))
        self.assertEqual(gate.shape, (self.B, self.T_a, 1))

    def test_gate_range(self):
        """Gate values nằm trong [0, 1]."""
        audio = torch.randn(self.B, self.T_a, self.D)
        text = torch.randn(self.B, self.T_t, self.D)

        _, gate = self.tme(audio, text)

        self.assertTrue(
            (gate >= 0.0).all() and (gate <= 1.0).all(),
            f"Gate values ngoài khoảng [0, 1]: min={gate.min()}, max={gate.max()}"
        )

    def test_with_masks(self):
        """Forward pass hoạt động đúng với attention masks."""
        audio = torch.randn(self.B, self.T_a, self.D)
        text = torch.randn(self.B, self.T_t, self.D)

        # Tạo mask: mỗi sample có độ dài khác nhau
        audio_mask = torch.ones(self.B, self.T_a, dtype=torch.bool)
        audio_mask[0, 40:] = False  # sample 0 ngắn hơn
        audio_mask[1, 30:] = False

        text_mask = torch.ones(self.B, self.T_t, dtype=torch.bool)
        text_mask[0, 8:] = False
        text_mask[2, 10:] = False

        enhanced, gate = self.tme(audio, text, audio_mask, text_mask)

        self.assertEqual(enhanced.shape, (self.B, self.T_a, self.D))
        self.assertEqual(gate.shape, (self.B, self.T_a, 1))

        # Gate tại vị trí padding audio phải = 0
        self.assertTrue(
            (gate[0, 40:] == 0.0).all(),
            "Gate phải = 0 tại vị trí padding audio"
        )
        self.assertTrue(
            (gate[1, 30:] == 0.0).all(),
            "Gate phải = 0 tại vị trí padding audio"
        )

    def test_gradient_flow(self):
        """Gradient chảy ngược qua TME."""
        self.tme.train()
        audio = torch.randn(self.B, self.T_a, self.D, requires_grad=True)
        text = torch.randn(self.B, self.T_t, self.D, requires_grad=True)

        enhanced, gate = self.tme(audio, text)
        loss = enhanced.sum() + gate.sum()
        loss.backward()

        self.assertIsNotNone(audio.grad)
        self.assertIsNotNone(text.grad)
        self.assertTrue(audio.grad.abs().sum() > 0, "Audio gradient = 0")
        self.assertTrue(text.grad.abs().sum() > 0, "Text gradient = 0")


class TestBiSequenceFusion(unittest.TestCase):
    """Tests cho BiSequenceFusion (fallback BiGRU)."""

    def setUp(self):
        self.B = 4
        self.T = 50
        self.D = 768
        # Luôn test BiGRU fallback vì mamba_ssm không có trên CPU
        self.fusion = BiSequenceFusion(
            hidden_dim=self.D, force_fallback=True
        )
        self.fusion.eval()

    def test_output_shape(self):
        """Pooled output có shape (B, 2*hidden_dim)."""
        x = torch.randn(self.B, self.T, self.D)
        pooled = self.fusion(x)
        expected_dim = self.D * 2  # mean + max concat
        self.assertEqual(pooled.shape, (self.B, expected_dim))

    def test_output_dim_attribute(self):
        """output_dim attribute khớp với shape thực tế."""
        x = torch.randn(self.B, self.T, self.D)
        pooled = self.fusion(x)
        self.assertEqual(pooled.shape[1], self.fusion.output_dim)

    def test_with_mask(self):
        """Forward pass hoạt động đúng với mask."""
        x = torch.randn(self.B, self.T, self.D)
        mask = torch.ones(self.B, self.T, dtype=torch.bool)
        mask[0, 30:] = False
        mask[1, 40:] = False

        pooled = self.fusion(x, mask=mask)
        expected_dim = self.D * 2
        self.assertEqual(pooled.shape, (self.B, expected_dim))

    def test_gradient_flow(self):
        """Gradient chảy ngược qua fusion."""
        self.fusion.train()
        x = torch.randn(self.B, self.T, self.D, requires_grad=True)

        pooled = self.fusion(x)
        loss = pooled.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(x.grad.abs().sum() > 0, "Gradient = 0")

    def test_using_mamba_flag(self):
        """force_fallback=True → using_mamba=False."""
        self.assertFalse(self.fusion.using_mamba)

    def test_different_hidden_dims(self):
        """Fusion hoạt động với hidden_dim khác 768."""
        for dim in [256, 384, 512]:
            fusion = BiSequenceFusion(hidden_dim=dim, force_fallback=True)
            x = torch.randn(2, 20, dim)
            pooled = fusion(x)
            self.assertEqual(pooled.shape, (2, dim * 2))


class TestFusionFallbackMock(unittest.TestCase):
    """
    Test nhánh fallback bằng cách mock ImportError cho mamba_ssm.
    Đảm bảo khi mamba_ssm không import được, BiSequenceFusion vẫn
    khởi tạo và chạy forward bình thường với BiGRU.
    """

    def test_fallback_when_mamba_unavailable(self):
        """
        Giả lập mamba_ssm không có → fusion phải dùng BiGRU
        và in warning.
        """
        import src.models.fusion as fusion_module

        # Lưu trạng thái gốc
        original_flag = fusion_module._MAMBA_AVAILABLE

        try:
            # Mock: giả lập mamba_ssm không có
            fusion_module._MAMBA_AVAILABLE = False

            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                fusion = BiSequenceFusion(
                    hidden_dim=768, force_fallback=False
                )
                # Phải có warning vì mamba không có
                mamba_warnings = [
                    x for x in w
                    if "mamba-ssm không có sẵn" in str(x.message)
                ]
                self.assertTrue(
                    len(mamba_warnings) > 0,
                    "Phải có warning khi mamba-ssm không có"
                )

            # Fusion vẫn phải chạy được
            self.assertFalse(fusion.using_mamba)
            x = torch.randn(2, 20, 768)
            pooled = fusion(x)
            self.assertEqual(pooled.shape, (2, 768 * 2))
        finally:
            # Khôi phục trạng thái gốc
            fusion_module._MAMBA_AVAILABLE = original_flag


class TestViMambaSERClassifier(unittest.TestCase):
    """Tests cho pipeline tổng hợp ViMambaSERClassifier."""

    def setUp(self):
        self.B = 4
        self.T_a = 50
        self.T_t = 12
        self.D = 768
        self.num_classes = 4

        self.model = ViMambaSERClassifier(
            embed_dim=self.D,
            num_classes=self.num_classes,
            tme_num_heads=8,
            fusion_hidden=self.D,
            fusion_force_fallback=True,  # CPU test → dùng fallback
            mlp_dropout=0.3,
        )

    def test_output_shape(self):
        """Logits shape đúng (B, num_classes)."""
        self.model.eval()
        audio = torch.randn(self.B, self.T_a, self.D)
        text = torch.randn(self.B, self.T_t, self.D)

        out = self.model(audio, text)

        self.assertEqual(out["logits"].shape, (self.B, self.num_classes))
        self.assertEqual(out["gate"].shape, (self.B, self.T_a, 1))

    def test_output_with_masks(self):
        """Pipeline chạy đúng với masks."""
        self.model.eval()
        audio = torch.randn(self.B, self.T_a, self.D)
        text = torch.randn(self.B, self.T_t, self.D)

        audio_mask = torch.ones(self.B, self.T_a, dtype=torch.bool)
        audio_mask[0, 40:] = False
        text_mask = torch.ones(self.B, self.T_t, dtype=torch.bool)
        text_mask[0, 8:] = False

        out = self.model(audio, text, audio_mask, text_mask)

        self.assertEqual(out["logits"].shape, (self.B, self.num_classes))

    def test_full_gradient_flow(self):
        """Gradient chảy ngược qua toàn bộ pipeline."""
        self.model.train()
        audio = torch.randn(self.B, self.T_a, self.D, requires_grad=True)
        text = torch.randn(self.B, self.T_t, self.D, requires_grad=True)

        out = self.model(audio, text)
        loss = nn.CrossEntropyLoss()(
            out["logits"],
            torch.randint(0, self.num_classes, (self.B,))
        )
        loss.backward()

        self.assertIsNotNone(audio.grad)
        self.assertIsNotNone(text.grad)
        self.assertTrue(audio.grad.abs().sum() > 0)
        self.assertTrue(text.grad.abs().sum() > 0)

        # Kiểm tra gradient chảy vào TME và classifier
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(
                    param.grad,
                    f"Gradient None cho parameter: {name}"
                )

    def test_gate_in_valid_range(self):
        """Gate từ pipeline tổng hợp vẫn nằm trong [0, 1]."""
        self.model.eval()
        audio = torch.randn(self.B, self.T_a, self.D)
        text = torch.randn(self.B, self.T_t, self.D)

        out = self.model(audio, text)
        gate = out["gate"]

        self.assertTrue(
            (gate >= 0.0).all() and (gate <= 1.0).all(),
            f"Gate ngoài [0,1]: min={gate.min():.4f}, max={gate.max():.4f}"
        )

    def test_with_projection(self):
        """Pipeline hoạt động khi fusion_hidden != embed_dim."""
        model = ViMambaSERClassifier(
            embed_dim=768,
            num_classes=4,
            fusion_hidden=384,  # khác embed_dim → cần projection
            fusion_force_fallback=True,
        )
        model.eval()

        audio = torch.randn(2, 30, 768)
        text = torch.randn(2, 10, 768)

        out = model(audio, text)
        self.assertEqual(out["logits"].shape, (2, 4))

    def test_variable_sequence_lengths(self):
        """
        Pipeline xử lý được các batch có sequence length khác nhau
        (qua padding + mask).
        """
        self.model.eval()
        # Batch với T_a, T_t cố định nhưng mask đa dạng
        audio = torch.randn(3, 60, self.D)
        text = torch.randn(3, 15, self.D)

        audio_mask = torch.ones(3, 60, dtype=torch.bool)
        audio_mask[0, 20:] = False   # sample 0: 20 frames
        audio_mask[1, 45:] = False   # sample 1: 45 frames
        # sample 2: 60 frames (full)

        text_mask = torch.ones(3, 15, dtype=torch.bool)
        text_mask[0, 5:] = False
        text_mask[1, 10:] = False

        out = self.model(audio, text, audio_mask, text_mask)
        self.assertEqual(out["logits"].shape, (3, self.num_classes))


class TestCacheEmbeddings(unittest.TestCase):
    """Tests cho hàm cache/load embedding."""

    def test_cache_and_load(self):
        """Cache → load lại đúng tensor."""
        import tempfile
        from src.features.extract_sequence_embeddings import (
            cache_embedding, load_cached_embedding,
        )

        emb = torch.randn(50, 768)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = cache_embedding(emb, "test_sample_001", tmpdir, "audio")
            self.assertTrue(path.exists())

            loaded = load_cached_embedding("test_sample_001", tmpdir, "audio")
            self.assertIsNotNone(loaded)
            self.assertTrue(torch.allclose(emb, loaded))

    def test_load_nonexistent(self):
        """Load sample chưa cache → None."""
        import tempfile
        from src.features.extract_sequence_embeddings import (
            load_cached_embedding,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_cached_embedding("nonexistent", tmpdir, "audio")
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
