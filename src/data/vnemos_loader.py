import os
from pathlib import Path

# Mapping folder name của VNEMOS (bản Midterm) sang LABEL_NAMES của Phase F
VNEMOS_LABEL_MAP = {
    "Happiness": "happy",
    "Neutral": "neutral",
    "Sadness": "sad",
    "Anger": "angry"
    # Cố tình bỏ qua "Anxiety" vì model đang được train với 4 lớp
}

def load_local_vnemos(data_dir="data/raw/audio", target_classes=None):
    """
    Load danh sách file audio từ dataset VNEMOS được lưu cục bộ.
    
    Args:
        data_dir: Đường dẫn thư mục gốc chứa các sub-folder cảm xúc.
        target_classes: Danh sách các lớp cần load (mặc định lấy 4 lớp trong VNEMOS_LABEL_MAP).
        
    Returns:
        samples (list[tuple]): Danh sách (filepath, label) cho pipeline extract_sequence_embeddings.
    """
    if target_classes is None:
        target_classes = list(VNEMOS_LABEL_MAP.values())
        
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục {data_path}")
        
    samples = []
    
    for class_folder in data_path.iterdir():
        if not class_folder.is_dir():
            continue
            
        folder_name = class_folder.name
        
        # Nếu thư mục này thuộc VNEMOS mapping
        if folder_name in VNEMOS_LABEL_MAP:
            mapped_label = VNEMOS_LABEL_MAP[folder_name]
            
            # Chỉ lấy nếu label nằm trong target_classes
            if mapped_label in target_classes:
                for audio_file in class_folder.glob("*.wav"):
                    samples.append((str(audio_file), mapped_label))
                    
    print(f"Đã load {len(samples)} mẫu từ VNEMOS (bỏ qua các lớp không thuộc {target_classes})")
    return samples

if __name__ == "__main__":
    # Test thử
    samples = load_local_vnemos()
    print("Mẫu thử:", samples[:3])
