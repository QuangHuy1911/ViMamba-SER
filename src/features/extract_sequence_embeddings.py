"""
Trích xuất sequence-level embeddings từ WavLM-Base và PhoBERT-v2.

Khác với giai đoạn giữa kỳ (mean-pool → vector 768-d), ở đây giữ
nguyên chiều thời gian: WavLM → (T_audio, 768), PhoBERT → (T_text, 768).

Mỗi sample được cache thành file .pt theo sample_id, để khi Colab
bị ngắt session giữa chừng thì không cần tính lại từ đầu.
"""

import os
from pathlib import Path
from typing import Optional

import torch
import numpy as np

# Các import nặng (transformers) chỉ cần khi thực sự trích đặc trưng,
# không cần khi chạy unit test với tensor giả.
_TRANSFORMERS_AVAILABLE = False
try:
    from transformers import (
        WavLMModel, AutoProcessor,
        AutoModel, AutoTokenizer,
    )
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


def download_visec_if_needed(raw_dir: str = "data/raw/audio"):
    """
    Tải dataset ViSEC từ HuggingFace nếu chưa có sẵn ở thư mục raw_dir.
    Dùng thư viện `datasets` để tải và trích xuất file .wav.
    """
    out_dir = Path(raw_dir)
    # Kiểm tra xem đã có data chưa (giả sử có ít nhất 1000 file thì coi như đã tải)
    if out_dir.exists() and len(list(out_dir.glob("*.wav"))) > 1000:
        print(f"Dataset đã tồn tại ở {raw_dir}, bỏ qua bước tải từ HuggingFace.")
        return

    print(f"Đang tự động tải dataset hustep-lab/ViSEC từ HuggingFace xuống {raw_dir}...")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from datasets import load_dataset
        import soundfile as sf
        
        # Load dataset từ HF, thông thường ViSEC nằm ở split 'train'
        dataset = load_dataset("hustep-lab/ViSEC", split="train")
        
        print(f"Đã tải xong metadata, tiến hành lưu {len(dataset)} file .wav...")
        for idx, item in enumerate(dataset):
            audio_data = item.get("audio")
            if audio_data is None:
                continue
                
            waveform = audio_data["array"]
            sr = audio_data["sampling_rate"]
            
            # Đặt tên file theo chuẩn của project: sample_00000.wav
            filename = f"sample_{idx:05d}.wav"
            out_path = out_dir / filename
            
            sf.write(str(out_path), waveform, sr)
            
        print("Tải và lưu ViSEC dataset thành công!")
    except ImportError:
        print("CẢNH BÁO: Chưa cài đặt thư viện 'datasets' hoặc 'soundfile'. Hãy chạy: pip install datasets soundfile")
    except Exception as e:
        print(f"Lỗi khi tải dataset từ HuggingFace: {e}")

def extract_wavlm_sequence(
    waveform: torch.Tensor,
    model: "WavLMModel",
    processor: "AutoProcessor",
    device: str = "cpu",
) -> torch.Tensor:
    """
    Trích chuỗi embedding từ WavLM-Base (last hidden state, không pool).

    Parameters
    ----------
    waveform  : (num_samples,) — audio waveform 16kHz, mono, float32
    model     : WavLMModel đã load (frozen)
    processor : AutoProcessor tương ứng
    device    : "cpu" hoặc "cuda"

    Returns
    -------
    embeddings : (T_audio, 768) — T_audio phụ thuộc độ dài audio
                 Với WavLM-Base, T_audio ≈ num_samples / 320
    """
    inputs = processor(
        waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform,
        sampling_rate=16000,
        return_tensors="pt",
        padding=False,
    )
    input_values = inputs["input_values"].to(device)

    with torch.no_grad():
        outputs = model(input_values)
        # last_hidden_state: (1, T_audio, 768)
        seq_embedding = outputs.last_hidden_state.squeeze(0).cpu()

    return seq_embedding  # (T_audio, 768)


def extract_phobert_sequence(
    transcript: str,
    model: "AutoModel",
    tokenizer: "AutoTokenizer",
    device: str = "cpu",
    max_length: int = 256,
) -> torch.Tensor:
    """
    Trích chuỗi embedding token-level từ PhoBERT-v2 cho 1 transcript.

    Parameters
    ----------
    transcript : str — transcript sinh từ PhoWhisper
    model      : PhoBERT-v2 model (frozen)
    tokenizer  : AutoTokenizer tương ứng
    device     : "cpu" hoặc "cuda"
    max_length : int — giới hạn chiều dài tokenize

    Returns
    -------
    embeddings : (T_text, 768) — T_text = số token thực (không kể padding,
                 nhưng bao gồm [CLS] và [SEP])
    """
    encoded = tokenizer(
        transcript,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state: (1, T_text, 768)
        seq_embedding = outputs.last_hidden_state.squeeze(0).cpu()

    return seq_embedding  # (T_text, 768)


def cache_embedding(
    embedding: torch.Tensor,
    sample_id: str,
    cache_dir: str,
    modality: str,
) -> Path:
    """
    Lưu embedding ra file .pt theo sample_id và modality.

    Tên file: {cache_dir}/{modality}/{sample_id}.pt
    Tạo thư mục tự động nếu chưa có.

    Parameters
    ----------
    embedding  : tensor cần lưu (T, D)
    sample_id  : ID duy nhất của sample (vd "sample_0001")
    cache_dir  : thư mục gốc chứa cache
    modality   : "audio" hoặc "text"

    Returns
    -------
    path : Path — đường dẫn file đã lưu
    """
    out_dir = Path(cache_dir) / modality
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sample_id}.pt"
    torch.save(embedding, out_path)
    return out_path


def load_cached_embedding(
    sample_id: str,
    cache_dir: str,
    modality: str,
) -> Optional[torch.Tensor]:
    """
    Load embedding đã cache. Trả về None nếu chưa có cache.
    """
    cache_path = Path(cache_dir) / modality / f"{sample_id}.pt"
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=True)
    return None


def extract_and_cache_all(
    samples: list[dict],
    wavlm_model,
    wavlm_processor,
    phobert_model,
    phobert_tokenizer,
    cache_dir: str,
    device: str = "cpu",
    skip_existing: bool = True,
) -> dict[str, dict[str, Path]]:
    """
    Trích xuất và cache embedding cho toàn bộ danh sách samples.

    Parameters
    ----------
    samples : list[dict] — mỗi dict có:
        - "sample_id": str
        - "waveform": torch.Tensor (num_samples,)
        - "transcript": str
    wavlm_model, wavlm_processor : WavLM-Base model + processor
    phobert_model, phobert_tokenizer : PhoBERT-v2 model + tokenizer
    cache_dir  : thư mục gốc cache
    device     : "cpu" hoặc "cuda"
    skip_existing : bỏ qua sample đã có cache (mặc định True)

    Returns
    -------
    paths : dict[sample_id → {"audio": Path, "text": Path}]
    """
    from tqdm import tqdm

    result = {}
    for sample in tqdm(samples, desc="Extracting sequence embeddings"):
        sid = sample["sample_id"]
        paths = {}

        # Audio embedding
        cached_audio = load_cached_embedding(sid, cache_dir, "audio")
        if cached_audio is not None and skip_existing:
            paths["audio"] = Path(cache_dir) / "audio" / f"{sid}.pt"
        else:
            audio_emb = extract_wavlm_sequence(
                sample["waveform"], wavlm_model, wavlm_processor, device
            )
            paths["audio"] = cache_embedding(audio_emb, sid, cache_dir, "audio")

        # Text embedding
        cached_text = load_cached_embedding(sid, cache_dir, "text")
        if cached_text is not None and skip_existing:
            paths["text"] = Path(cache_dir) / "text" / f"{sid}.pt"
        else:
            text_emb = extract_phobert_sequence(
                sample["transcript"], phobert_model, phobert_tokenizer, device
            )
            paths["text"] = cache_embedding(text_emb, sid, cache_dir, "text")

        result[sid] = paths

    return result
