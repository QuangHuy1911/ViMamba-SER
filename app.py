import gradio as gr
import torch
import librosa
import numpy as np
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers import AutoTokenizer, AutoModel
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from src.models.vimamba_ser import ViMambaSERClassifier
from src.config import LABEL_NAMES, NUM_CLASSES
import os

print("Đang tải các mô hình nền tảng (chỉ tải 1 lần)...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Tải PhoWhisper
whisper_processor = AutoProcessor.from_pretrained("vinai/phowhisper-base")
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained("vinai/phowhisper-base").to(device)

# 2. Tải PhoBERT
phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
phobert_model = AutoModel.from_pretrained("vinai/phobert-base-v2").to(device)

# 3. Tải WavLM
wavlm_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-base").to(device)

# 4. Tải Model ViMamba-SER (Não bộ Demo)
# LƯU Ý CHO ĐẠT: Kiểm tra xem file f_seed42_best.pt có nằm đúng trong thư mục runs/ không nhé!
checkpoint_path = "runs/f_seed42_best.pt"
if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    config_dict = ckpt.get('config', {})
    
    # Khởi tạo model giống lúc train Phase F (với hidden_dim=256)
    model = ViMambaSERClassifier(
        embed_dim=768,
        num_classes=NUM_CLASSES,
        fusion_hidden=256,
        fusion_force_fallback=True
    ).to(device)
    
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("Đã tải xong Model ViMamba-SER (BiGRU Fallback)!")
else:
    print(f"CẢNH BÁO: Không tìm thấy file {checkpoint_path}. Hãy chạy lệnh train full trước khi mở web!")
    model = None

def process_audio(audio_path):
    if audio_path is None or model is None:
        return "Lỗi: Chưa upload âm thanh hoặc Model chưa sẵn sàng.", {}

    # Load audio 16kHz
    speech, sr = librosa.load(audio_path, sr=16000)
    
    # 1. PhoWhisper: Lấy Text
    input_features = whisper_processor(speech, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    predicted_ids = whisper_model.generate(input_features)
    transcript = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    # 2. PhoBERT: Trích xuất Text Sequence (1, T_t, 768)
    text_inputs = phobert_tokenizer(transcript, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    with torch.no_grad():
        text_outputs = phobert_model(**text_inputs)
        text_seq = text_outputs.last_hidden_state  # (1, T_t, 768)
        text_mask = text_inputs.attention_mask.bool()
        
    # 3. WavLM: Trích xuất Audio Sequence (1, T_a, 768)
    audio_inputs = wavlm_extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        audio_outputs = wavlm_model(audio_inputs.input_values)
        audio_seq = audio_outputs.last_hidden_state  # (1, T_a, 768)
        audio_mask = torch.ones((1, audio_seq.size(1)), dtype=torch.bool).to(device)
        
    # 4. Dự đoán qua ViMamba-SER
    with torch.no_grad():
        outputs = model(audio_seq, text_seq, audio_mask=audio_mask, text_mask=text_mask)
        logits = outputs['logits']
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        
    # Tạo dictionary kết quả
    results = {LABEL_NAMES[i].capitalize(): float(probs[i]) for i in range(NUM_CLASSES)}
    
    return transcript, results

# Giao diện Gradio
demo = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath", label="Tải lên hoặc Ghi âm"),
    outputs=[
        gr.Textbox(label="Văn bản nhận dạng (PhoWhisper)"),
        gr.Label(num_top_classes=4, label="Cảm xúc dự đoán")
    ],
    title="🎙️ ViMamba-SER: Nhận dạng Cảm xúc Giọng nói Tiếng Việt",
    description="Hệ thống kết hợp Âm thanh (WavLM) và Văn bản (PhoBERT) qua cơ chế Text-aware Modality Enhancement."
)

if __name__ == "__main__":
    demo.launch(share=True)
