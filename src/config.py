# ============================================================
# config.py — Single source of truth
# Mọi notebook đều import từ đây, không hardcode giá trị
# ============================================================
import torch, os

SEED            = 42
DATASET_NAME    = "hustep-lab/ViSEC"
LABEL_MAP       = {"happy": 0, "neutral": 1, "sad": 2, "angry": 3}
LABEL_NAMES     = ["happy", "neutral", "sad", "angry"]
NUM_CLASSES     = 4

WAVLM_CKPT      = "microsoft/wavlm-base"
PHOWHISPER_CKPT = "vinai/phowhisper-base"
PHOBERT_CKPT    = "vinai/phobert-base-v2"

SAMPLE_RATE     = 16000
AUDIO_DIM       = 768
TEXT_DIM        = 768
FUSED_DIM       = AUDIO_DIM + TEXT_DIM

P2_HIDDEN_DIM   = 256
P2_DROPOUT      = 0.3
P2_LR           = 1e-3
P2_EPOCHS       = 30
P2_BATCH_SIZE   = 64

P3_HIDDEN_DIM   = 512
P3_DROPOUT      = 0.3
P3_LR           = 1e-3
P3_EPOCHS       = 30
P3_BATCH_SIZE   = 64

EMBED_DIR       = "../data/embeddings"
RUNS_DIR        = "../runs"
FIGURES_DIR     = "../reports/figures"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

for d in [EMBED_DIR, RUNS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)
