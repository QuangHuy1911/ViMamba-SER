"""
README: ViMamba-SER Gradio Demo
-------------------------------
Hướng dẫn chạy:
1. Đảm bảo bạn đang ở thư mục gốc của repo: `cd ViMamba-SER`
2. Cài đặt các thư viện cần thiết: `pip install -r app/requirements.txt`
3. Chạy ứng dụng: `python app/demo_app.py`

Yêu cầu về Checkpoint (sẽ tự động tạo model ngẫu nhiên nếu không tìm thấy để demo giao diện):
- Đặt checkpoint Phase F vào: `runs/phase_f_best.pt` (hoặc cấu hình đường dẫn qua biến ở dưới)
- Đặt checkpoint Phase A vào: `runs/phase_a_best.pt`
Các checkpoint này phải được lưu bằng `src.utils.checkpoint.save_checkpoint`.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torchaudio
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from transformers import WavLMModel, AutoProcessor, AutoModel, AutoTokenizer
from transformers import WhisperProcessor, WhisperForConditionalGeneration

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import DEVICE, SAMPLE_RATE, LABEL_NAMES, WAVLM_CKPT, PHOWHISPER_CKPT, PHOBERT_CKPT
from src.models.vimamba_ser import ViMambaSERClassifier
from src.features.extract_sequence_embeddings import extract_wavlm_sequence, extract_phobert_sequence
from src.utils.checkpoint import load_checkpoint
from train import build_mlp

# Cấu hình đường dẫn checkpoint
PHASE_F_CKPT_PATH = "runs/phase_f_best.pt"
PHASE_A_CKPT_PATH = "runs/phase_a_best.pt"

# Map cảm xúc sang Emoji
EMOJI_MAP = {
    "happy": "😄 Vui vẻ (Happy)",
    "neutral": "😐 Bình thường (Neutral)",
    "sad": "😢 Buồn bã (Sad)",
    "angry": "😡 Tức giận (Angry)"
}

print("Đang tải các mô hình trích xuất đặc trưng (có thể mất vài phút lần đầu)...")
# Load Frozen Encoders
wavlm_processor = AutoProcessor.from_pretrained(WAVLM_CKPT)
wavlm_model = WavLMModel.from_pretrained(WAVLM_CKPT).to(DEVICE).eval()

whisper_processor = WhisperProcessor.from_pretrained(PHOWHISPER_CKPT)
whisper_model = WhisperForConditionalGeneration.from_pretrained(PHOWHISPER_CKPT).to(DEVICE).eval()

phobert_tokenizer = AutoTokenizer.from_pretrained(PHOBERT_CKPT)
phobert_model = AutoModel.from_pretrained(PHOBERT_CKPT).to(DEVICE).eval()

print("Hoàn tất tải mô hình trích xuất đặc trưng!")

def load_phase_f_model():
    model = ViMambaSERClassifier(fusion_force_fallback=True) # Fallback to BiGRU if Mamba not installed
    if os.path.exists(PHASE_F_CKPT_PATH):
        print(f"Đang load checkpoint Phase F từ {PHASE_F_CKPT_PATH}...")
        load_checkpoint(PHASE_F_CKPT_PATH, model=model, device=DEVICE)
    else:
        print(f"CẢNH BÁO: Không tìm thấy checkpoint Phase F tại {PHASE_F_CKPT_PATH}. Đang dùng model ngẫu nhiên để demo giao diện.")
    model.to(DEVICE).eval()
    return model

def load_phase_a_model():
    model = build_mlp(input_dim=768, num_classes=4)
    if os.path.exists(PHASE_A_CKPT_PATH):
        print(f"Đang load checkpoint Phase A từ {PHASE_A_CKPT_PATH}...")
        load_checkpoint(PHASE_A_CKPT_PATH, model=model, device=DEVICE)
    else:
        print(f"CẢNH BÁO: Không tìm thấy checkpoint Phase A tại {PHASE_A_CKPT_PATH}. Đang dùng model ngẫu nhiên để demo giao diện.")
    model.to(DEVICE).eval()
    return model

phase_f_model = load_phase_f_model()
phase_a_model = load_phase_a_model()

def process_audio(audio_path, model_type="Phase F (TME + Fusion)"):
    start_time = time.time()
    
    if audio_path is None:
        return "Lỗi: Không có âm thanh", None, "Vui lòng thu âm hoặc tải lên một file.", "0 ms"
        
    try:
        waveform, sr = torchaudio.load(audio_path)
        if waveform.size(1) / sr < 0.3:
            return "Lỗi: Âm thanh quá ngắn", None, "Vui lòng cung cấp đoạn âm thanh dài hơn 0.3 giây.", "0 ms"
            
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)
            
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        waveform_np = waveform.squeeze().numpy()
        
        with torch.no_grad():
            # Bước 1: Trích xuất WavLM embedding
            audio_seq = extract_wavlm_sequence(waveform, wavlm_model, wavlm_processor, DEVICE)
            
            if model_type == "Phase A (Audio-only)":
                audio_emb_mean = audio_seq.mean(dim=0, keepdim=True)
                logits = phase_a_model(audio_emb_mean)
                transcript = "(Phase A không sử dụng transcript)"
            else:
                # Bước 2: Sinh transcript bằng PhoWhisper
                whisper_inputs = whisper_processor(waveform_np, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(DEVICE)
                generated_ids = whisper_model.generate(whisper_inputs)
                transcript = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                if not transcript.strip():
                    transcript = " "
                
                # Bước 3: Trích xuất PhoBERT embedding
                text_seq = extract_phobert_sequence(transcript, phobert_model, phobert_tokenizer, DEVICE)
                
                # Bước 4: Dự đoán cảm xúc (Phase F)
                outputs = phase_f_model(audio_seq.unsqueeze(0), text_seq.unsqueeze(0))
                logits = outputs["logits"]
                
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        pred_idx = probs.argmax()
        pred_label = LABEL_NAMES[pred_idx]
        
        # Vẽ biểu đồ
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar([EMOJI_MAP[l] for l in LABEL_NAMES], probs, color=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Xác suất')
        ax.set_title('Phân bố xác suất các lớp cảm xúc')
        plt.xticks(rotation=15)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval*100:.1f}%', ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        
        end_time = time.time()
        process_ms = f"{(end_time - start_time)*1000:.0f} ms"
        
        return EMOJI_MAP[pred_label], fig, transcript, process_ms
        
    except Exception as e:
        return f"Lỗi hệ thống: {str(e)}", None, "Đã xảy ra lỗi trong quá trình xử lý.", "0 ms"

# Giao diện Gradio
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
)

with gr.Blocks(theme=theme, title="ViMamba-SER Demo") as demo:
    gr.Markdown("# 🎙️ ViMamba-SER: Nhận Dạng Cảm Xúc Giọng Nói Tiếng Việt")
    gr.Markdown("Đồ án môn Speech Processing (SLP) - Mô hình nhận dạng 4 lớp cảm xúc: Vui vẻ, Bình thường, Buồn bã, Tức giận.")
    
    with gr.Tabs():
        with gr.TabItem("Phase F (TME + Fusion)"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Đầu vào Âm thanh")
                    analyze_btn = gr.Button("Phân tích cảm xúc", variant="primary", size="lg")
                    
                    gr.Markdown("### Audio Mẫu (Từ tập ViSEC)")
                    examples = gr.Examples(
                        examples=[
                            ["data/raw/audio/Anger/anger_cua_lai_vo_bau_01.wav"],
                            ["data/raw/audio/Happiness/happiness_chang_trai_nam_ay_01.wav"],
                            ["data/raw/audio/Neutral/song_o_day_song_tap1_neutral_01.wav"],
                            ["data/raw/audio/Sadness/bassihanhphuc_00-07-27_to_00-07-37.wav"]
                        ],
                        inputs=audio_input,
                        label="Chọn một mẫu có sẵn để thử nghiệm:"
                    )
                
                with gr.Column(scale=1):
                    label_output = gr.Textbox(label="Kết quả Dự đoán", text_align="center", scale=2)
                    plot_output = gr.Plot(label="Biểu đồ Xác suất")
                    transcript_output = gr.Textbox(label="Transcript (PhoWhisper sinh tự động)", info="*Text chỉ mang tính tham khảo, mô hình chủ yếu dựa vào audio.")
                    time_output = gr.Textbox(label="Thời gian xử lý")
                    
            analyze_btn.click(
                fn=lambda x: process_audio(x, "Phase F (TME + Fusion)"),
                inputs=[audio_input],
                outputs=[label_output, plot_output, transcript_output, time_output]
            )
            
        with gr.TabItem("So sánh mô hình (Phase A vs Phase F)"):
            gr.Markdown("So sánh trực tiếp kết quả của mô hình Audio-only (Phase A) và mô hình đa phương thức Text-enhanced (Phase F).")
            with gr.Row():
                compare_audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Âm thanh đầu vào")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Phase A (Chỉ dùng Audio)")
                    a_label = gr.Textbox(label="Dự đoán (Phase A)")
                    a_plot = gr.Plot(label="Xác suất (Phase A)")
                    a_time = gr.Textbox(label="Thời gian xử lý")
                with gr.Column():
                    gr.Markdown("### Phase F (Audio + Text + TME + Fusion)")
                    f_label = gr.Textbox(label="Dự đoán (Phase F)")
                    f_plot = gr.Plot(label="Xác suất (Phase F)")
                    f_transcript = gr.Textbox(label="Transcript")
                    f_time = gr.Textbox(label="Thời gian xử lý")
                    
            compare_btn = gr.Button("So sánh 2 mô hình", variant="primary")
            
            def compare_both(audio):
                a_lbl, a_plt, _, a_tm = process_audio(audio, "Phase A (Audio-only)")
                f_lbl, f_plt, f_txt, f_tm = process_audio(audio, "Phase F (TME + Fusion)")
                return a_lbl, a_plt, a_tm, f_lbl, f_plt, f_txt, f_tm
                
            compare_btn.click(
                fn=compare_both,
                inputs=[compare_audio],
                outputs=[a_label, a_plot, a_time, f_label, f_plot, f_transcript, f_time]
            )

if __name__ == "__main__":
    print("Khởi động Gradio server...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
