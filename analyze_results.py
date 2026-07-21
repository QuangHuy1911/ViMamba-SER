import argparse
import os
import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import LABEL_MAP, LABEL_NAMES, NUM_CLASSES, DEVICE
from src.models.vimamba_ser import ViMambaSERClassifier
from src.utils.checkpoint import load_checkpoint
from train import SequenceDataset, collate_fn_seq, set_seed

def main(args):
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("--- Phân tích kết quả Phase F ---")
    
    # 1. Load data & test split
    labels_candidates = [
        os.path.join("runs", "midterm_results", "labels.npy"),
        os.path.join(args.embed_dir, "labels.npy"),
        os.path.join(args.embed_dir, "sequence", "labels.npy"),
    ]
    labels_path = next((lp for lp in labels_candidates if os.path.exists(lp)), None)
    if not labels_path:
        raise FileNotFoundError(f"Không tìm thấy labels.npy ở: {labels_candidates}")
        
    labels = np.load(labels_path, allow_pickle=True)
    if labels.dtype.kind in ('U', 'S', 'O'):
        labels = np.array([LABEL_MAP[str(l)] for l in labels], dtype=np.int64)
        
    all_indices = np.arange(len(labels))
    
    # Tái tạo đúng logic split của test set
    if args.fold is None:
        if args.data_split and os.path.exists(args.data_split):
            with open(args.data_split, 'r') as f:
                splits = json.load(f)
            test_idx = splits['test']
        else:
            raise FileNotFoundError("data_split json not found and fold is None.")
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        folds = list(skf.split(all_indices, labels))
        _, test_idx = folds[args.fold]
        
    print(f"Test samples: {len(test_idx)}")
    
    cache_dir = os.path.join(args.embed_dir, "sequence")
    test_dataset = SequenceDataset(test_idx, labels[test_idx], cache_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_seq)
    
    # 2. Khởi tạo mô hình
    model = ViMambaSERClassifier(fusion_force_fallback=args.force_fallback)
    
    label_names = LABEL_NAMES # Default to config
    
    # Thử tải checkpoint, nếu không có thì khởi tạo random weights để demo
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint từ {args.checkpoint}...")
        ckpt = load_checkpoint(args.checkpoint, model=model, device=DEVICE)
        if ckpt and 'label_encoder' in ckpt and 'LABEL_NAMES' in ckpt['label_encoder']:
            label_names = ckpt['label_encoder']['LABEL_NAMES']
    else:
        print(f"CẢNH BÁO: Không tìm thấy checkpoint tại {args.checkpoint}. Đang dùng weights ngẫu nhiên (Mock mode)!")
        
    print(f"THỨ TỰ NHÃN ĐANG DÙNG ĐỂ VẼ: {label_names}")
        
    model.to(DEVICE)
    model.eval()
    
    # 3. Chạy inference
    all_preds = []
    all_labels = []
    all_embeddings = []
    
    print("Đang chạy inference trên tập test...")
    try:
        with torch.no_grad():
            for batch in test_loader:
                a_seq, t_seq, a_mask, t_mask, y = batch
                a_seq, t_seq = a_seq.to(DEVICE), t_seq.to(DEVICE)
                a_mask, t_mask = a_mask.to(DEVICE), t_mask.to(DEVICE)
                
                # Forward pass thủ công để lấy embedding trước lớp classifier
                enhanced, _ = model.tme(a_seq, t_seq, audio_mask=a_mask, text_mask=t_mask)
                if model.proj is not None:
                    enhanced = model.proj(enhanced)
                pooled = model.fusion(enhanced, mask=a_mask)
                logits = model.classifier(pooled)
                
                preds = logits.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.numpy())
                all_embeddings.append(pooled.cpu().numpy())
    except FileNotFoundError as e:
        print(f"CẢNH BÁO: Bỏ qua inference thật vì thiếu file embeddings ({e}). Đang tạo dữ liệu giả để test script vẽ hình...")
        # Tạo mock data
        num_mock_samples = len(test_idx)
        all_labels = labels[test_idx]
        all_preds = np.random.randint(0, NUM_CLASSES, size=num_mock_samples)
        # pooled shape (num_samples, 2*fusion_hidden)
        # giả sử fusion_hidden = 768
        mock_pooled = np.random.randn(num_mock_samples, 1536)
        all_embeddings = [mock_pooled]
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_embeddings = np.vstack(all_embeddings)
    
    acc = (all_preds == all_labels).mean()
    print(f"Test Accuracy: {acc:.4f}")
    
    # 4. Vẽ Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix - Phase F')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(args.out_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Đã lưu Confusion Matrix tại {cm_path}")
    
    # 5. Vẽ t-SNE
    print("Đang tính toán t-SNE...")
    tsne = TSNE(n_components=2, random_state=args.seed, init='pca', learning_rate='auto')
    embeds_2d = tsne.fit_transform(all_embeddings)
    
    plt.figure(figsize=(10, 8))
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'] # Happy, Neutral, Sad, Angry
    for i, label in enumerate(label_names):
        idx = (all_labels == i)
        plt.scatter(embeds_2d[idx, 0], embeds_2d[idx, 1], c=colors[i], label=label, alpha=0.7, edgecolors='w', s=50)
    plt.title('t-SNE Visualization of Sequence Fusion Embeddings (Phase F)')
    plt.legend()
    plt.tight_layout()
    tsne_path = os.path.join(args.out_dir, "tsne_visualization.png")
    plt.savefig(tsne_path, dpi=300)
    plt.close()
    print(f"Đã lưu t-SNE tại {tsne_path}")
    
    print("Hoàn tất phân tích.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="runs/phase_f_best.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=None, help="Chỉ định fold nếu train bằng CV")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embed_dir", type=str, default="data/embeddings")
    parser.add_argument("--out_dir", type=str, default="results/phase_f")
    parser.add_argument("--data_split", type=str, default="data/embeddings/splits.json")
    parser.add_argument("--force_fallback", action="store_true")
    
    args = parser.parse_args()
    main(args)
