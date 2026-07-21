import argparse
import os
import json
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from src.utils.checkpoint import load_checkpoint
from src.models.vimamba_ser import ViMambaSERClassifier
from src.config import NUM_CLASSES, LABEL_NAMES
from train import SequenceAudioTextDataset, seq_collate_fn
from torch.utils.data import DataLoader, Subset

def load_checkpoint_and_eval(checkpoint_path, phase, fold, seed, embed_dir, save_dir):
    print(f"=== EVAL TỪ CHECKPOINT ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Phase: {phase} | Fold: {fold} | Seed: {seed}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")
    
    # Load metadata
    ckpt = load_checkpoint(checkpoint_path, device=device)
    config_dict = ckpt.get('config', {})
    
    # Prepare Model
    if phase == 'f':
        model = ViMambaSERClassifier(
            audio_dim=config_dict.get('audio_dim', 768),
            text_dim=config_dict.get('text_dim', 768),
            hidden_dim=config_dict.get('hidden_dim', 256),
            num_classes=NUM_CLASSES,
            num_layers=config_dict.get('num_layers', 2),
            force_fallback=config_dict.get('force_fallback', True)
        )
    else:
        import torch.nn as nn
        model = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_CLASSES)
        )
        
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load Data (Tái hiện logic lấy test_idx từ train.py)
    labels = np.load(os.path.join(embed_dir, 'labels.npy'), allow_pickle=True)
    if labels.dtype.kind in {'U', 'S', 'O'} and isinstance(labels[0], str):
        from src.config import LABEL_MAP
        labels = np.array([LABEL_MAP[l] for l in labels])
        
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    splits = list(skf.split(np.zeros(len(labels)), labels))
    _, test_idx = splits[fold]
    
    print(f"Số lượng sample trong test set của Fold {fold}: {len(test_idx)}")
    
    # Dataset
    if phase == 'f':
        dataset = SequenceAudioTextDataset(embed_dir, labels)
        test_dataset = Subset(dataset, test_idx)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=seq_collate_fn)
    else:
        # Phase a/b
        wavlm = np.load(os.path.join(embed_dir, 'wavlm_embeddings.npy'))
        if phase == 'b':
            phobert = np.load(os.path.join(embed_dir, 'phobert_embeddings.npy'))
            feats = 0.9 * wavlm + 0.1 * phobert
        else:
            feats = wavlm
            
        from torch.utils.data import TensorDataset
        test_dataset = TensorDataset(torch.tensor(feats[test_idx], dtype=torch.float32), 
                                     torch.tensor(labels[test_idx], dtype=torch.long))
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
    # Inference
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    
    with torch.no_grad():
        for batch in test_loader:
            if phase in ['a', 'b']:
                x, y = batch
                x, y = x.to(device), y.to(device)
                logits = model(x)
            else:
                a_seq, t_seq, a_mask, t_mask, y = batch
                a_seq, t_seq = a_seq.to(device), t_seq.to(device)
                a_mask, t_mask = a_mask.to(device), t_mask.to(device)
                y = y.to(device)
                outputs = model(a_seq, t_seq, audio_mask=a_mask, text_mask=t_mask)
                logits = outputs['logits']
                
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_y_true.extend(y.cpu().numpy())
            all_y_pred.extend(preds.cpu().numpy())
            all_y_prob.extend(probs.cpu().numpy())
            
    # Lưu kết quả
    os.makedirs(save_dir, exist_ok=True)
    fold_str = f"fold{fold}"
    
    # 1. CSV
    res_df = pd.DataFrame({'y_true': all_y_true, 'y_pred': all_y_pred})
    for i in range(NUM_CLASSES):
        res_df[f'prob_class_{i}'] = np.array(all_y_prob)[:, i]
    out_csv = os.path.join(save_dir, f"predictions_phase{phase}_{fold_str}.csv")
    res_df.to_csv(out_csv, index=False)
    
    # 2. CM
    cm = confusion_matrix(all_y_true, all_y_pred)
    cm_path = os.path.join(save_dir, f"cm_phase{phase}_{fold_str}.npy")
    np.save(cm_path, cm)
    
    # 3. CR
    cr = classification_report(all_y_true, all_y_pred, target_names=LABEL_NAMES, output_dict=True, zero_division=0)
    cr_path = os.path.join(save_dir, f"classification_report_phase{phase}_{fold_str}.json")
    with open(cr_path, 'w', encoding='utf-8') as f:
        json.dump(cr, f, indent=4)
        
    print(f"Đã lưu kết quả đánh giá (Metric re-calculate) tại: {save_dir}")
    print("Classification Report:")
    print(classification_report(all_y_true, all_y_pred, target_names=LABEL_NAMES, zero_division=0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn file .pt")
    parser.add_argument("--phase", type=str, required=True, choices=['a', 'b', 'f'])
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embed_dir", type=str, default="data/embeddings")
    parser.add_argument("--save_dir", type=str, default="runs")
    
    args = parser.parse_args()
    load_checkpoint_and_eval(args.checkpoint, args.phase, args.fold, args.seed, args.embed_dir, args.save_dir)
