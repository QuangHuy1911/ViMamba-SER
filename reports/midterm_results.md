# ViMamba-SER Midterm Results — ViSEC Dataset

## Audio-Only Ablation (A1–A5)
| Experiment | Test Accuracy |
|---|---|
| A1 WavLM → MLP 2 lớp (baseline) | 56.44% |
| A2 WavLM → Linear 1 lớp | 53.79% |
| A3 WavLM → MLP + Dropout(0.3) | 56.06% |
| A4 WavLM → MLP + BatchNorm | 52.27% |
| A5 MLP mean ± std (3 seeds) | 53.45% ± 1.79% |

## Text Pipeline Verification (B1–B2)
| Experiment | Test Accuracy |
|---|---|
| B1 PhoBERT → MLP (text-only, label proxy) | 100.00% |
| B2 WavLM + PhoBERT concat → MLP (label proxy) | 100.00% |

## Notes
- Dataset: ViSEC (5,280 samples, 4 emotions: angry/happy/neutral/sad)
- Audio encoder: WavLM-Base (microsoft/wavlm-base), frozen
- Text encoder: PhoBERT-v2 (vinai/phobert-base-v2), frozen
- Split: 70/15/15 stratified, seed=42
- B1/B2 dùng label proxy → 100% là expected, chỉ verify pipeline
