import argparse
import os
import json
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

from src.config import LABEL_MAP, LABEL_NAMES, NUM_CLASSES, EMBED_DIR, RUNS_DIR
from src.utils.checkpoint import save_checkpoint
from src.models.vimamba_ser import ViMambaSERClassifier

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class VectorDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class SequenceDataset(Dataset):
    def __init__(self, indices, labels, cache_dir):
        self.indices = indices
        self.labels = labels
        self.cache_dir = cache_dir
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        sample_idx = self.indices[idx]
        label = self.labels[idx]
        
        # Load from cache
        sample_id = f"sample_{sample_idx:05d}"
        audio_path = os.path.join(self.cache_dir, "audio", f"{sample_id}.pt")
        text_path = os.path.join(self.cache_dir, "text", f"{sample_id}.pt")
        
        audio_seq = torch.load(audio_path, weights_only=True)
        text_seq = torch.load(text_path, weights_only=True)
        
        return audio_seq, text_seq, label

def collate_fn_seq(batch):
    audio_seqs, text_seqs, labels = zip(*batch)
    
    # Pad audio
    audio_lens = [seq.size(0) for seq in audio_seqs]
    max_audio_len = max(audio_lens)
    padded_audio = torch.zeros(len(batch), max_audio_len, audio_seqs[0].size(-1))
    audio_mask = torch.zeros(len(batch), max_audio_len, dtype=torch.bool)
    
    for i, seq in enumerate(audio_seqs):
        length = seq.size(0)
        padded_audio[i, :length] = seq
        audio_mask[i, :length] = True
        
    # Pad text
    text_lens = [seq.size(0) for seq in text_seqs]
    max_text_len = max(text_lens)
    padded_text = torch.zeros(len(batch), max_text_len, text_seqs[0].size(-1))
    text_mask = torch.zeros(len(batch), max_text_len, dtype=torch.bool)
    
    for i, seq in enumerate(text_seqs):
        length = seq.size(0)
        padded_text[i, :length] = seq
        text_mask[i, :length] = True
        
    return padded_audio, padded_text, audio_mask, text_mask, torch.tensor(labels, dtype=torch.long)

def build_mlp(input_dim=768, num_classes=4):
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )

def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"--- Training Phase {args.phase.upper()} ---")
    
    # Load labels — tìm theo thứ tự ưu tiên
    labels_candidates = [
        os.path.join("runs", "midterm_results", "labels.npy"),
        os.path.join(args.embed_dir, "labels.npy"),
        os.path.join(args.embed_dir, "sequence", "labels.npy"),
    ]
    labels_path = None
    for lp in labels_candidates:
        if os.path.exists(lp):
            labels_path = lp
            break
    if labels_path is None:
        raise FileNotFoundError(
            f"Không tìm thấy labels.npy ở: {labels_candidates}"
        )
    print(f"Labels: {labels_path}")
    
    labels = np.load(labels_path, allow_pickle=True)
    # Convert string labels → int nếu cần (labels midterm lưu dạng string)
    if labels.dtype.kind in ('U', 'S', 'O'):  # unicode, byte string, object
        labels = np.array([LABEL_MAP[str(l)] for l in labels], dtype=np.int64)
    num_samples = len(labels)
    all_indices = np.arange(num_samples)
    
    # Train/Val/Test Split
    if args.fold is None:
        if args.data_split and os.path.exists(args.data_split):
            with open(args.data_split, 'r') as f:
                splits = json.load(f)
            train_idx = splits['train']
            val_idx = splits['val']
            test_idx = splits['test']
        else:
            raise FileNotFoundError("data_split json not found and fold is None.")
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        folds = list(skf.split(all_indices, labels))
        train_val_idx, test_idx = folds[args.fold]
        
        # Split train_val thành train + val (85/15 stratified)
        from sklearn.model_selection import train_test_split
        labels_train_val = labels[train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.15,
            random_state=args.seed, stratify=labels_train_val,
        )
        
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Load data and prepare loaders
    if args.phase in ['a', 'b']:
        wavlm_path = os.path.join(args.embed_dir, "wavlm_embeddings.npy")
        wavlm_emb = np.load(wavlm_path)
        
        if args.phase == 'a':
            X = wavlm_emb
        else:
            phobert_path = os.path.join(args.embed_dir, "phobert_embeddings.npy")
            phobert_emb = np.load(phobert_path)
            X = 0.9 * wavlm_emb + 0.1 * phobert_emb
            
        train_dataset = VectorDataset(X[train_idx], labels[train_idx])
        val_dataset = VectorDataset(X[val_idx], labels[val_idx])
        test_dataset = VectorDataset(X[test_idx], labels[test_idx])
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        model = build_mlp(input_dim=768, num_classes=NUM_CLASSES)
        
    elif args.phase == 'f':
        cache_dir = os.path.join(args.embed_dir, "sequence")
        train_dataset = SequenceDataset(train_idx, labels[train_idx], cache_dir)
        val_dataset = SequenceDataset(val_idx, labels[val_idx], cache_dir)
        test_dataset = SequenceDataset(test_idx, labels[test_idx], cache_dir)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_seq)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_seq)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_seq)
        
        # Load config from phase_f.yaml
        config_f = {}
        try:
            with open("configs/phase_f.yaml", "r") as f:
                config_f = yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load phase_f.yaml. Using defaults. {e}")
            
        tme_args = config_f.get('tme', {})
        fusion_args = config_f.get('fusion', {})
        classifier_args = config_f.get('classifier', {})
        
        model = ViMambaSERClassifier(
            embed_dim=config_f.get('encoders', {}).get('embed_dim', 768),
            num_classes=NUM_CLASSES,
            tme_num_heads=tme_args.get('num_heads', 8),
            tme_dropout=tme_args.get('dropout', 0.1),
            fusion_hidden=fusion_args.get('hidden_dim', 768),
            fusion_force_fallback=args.force_fallback or fusion_args.get('force_fallback', False),
            mlp_dropout=classifier_args.get('dropout', 0.3),
            mamba_d_state=fusion_args.get('mamba_d_state', 16),
            mamba_d_conv=fusion_args.get('mamba_d_conv', 4),
            mamba_expand=fusion_args.get('mamba_expand', 2),
        )
    else:
        raise ValueError(f"Invalid phase: {args.phase}")
        
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_epoch = 0
    
    config_dict = {
        'phase': args.phase,
        'architecture': model.__class__.__name__,
        'force_fallback': args.force_fallback,
        'args': vars(args)
    }
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            if args.phase in ['a', 'b']:
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
                
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)
            
        # Eval
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                if args.phase in ['a', 'b']:
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
                    
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
                
        val_acc = val_correct / val_total
        print(f"Epoch {epoch:02d} | Train Acc: {train_correct/train_total:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                config_dict=config_dict,
                epoch=epoch,
                best_val_acc=best_val_acc,
                save_dir=args.save_dir,
                phase=args.phase,
                fold=args.fold,
                seed=args.seed
            )
            
    # Load best and test
    print(f"Loading best checkpoint from epoch {best_epoch} (Val Acc: {best_val_acc:.4f})")
    if args.fold is not None:
        ckpt_path = os.path.join(args.save_dir, f"{args.phase}_fold{args.fold}_best.pt")
    else:
        ckpt_path = os.path.join(args.save_dir, f"{args.phase}_seed{args.seed}_best.pt")
        
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    test_correct = 0
    test_total = 0
    class_correct = {i: 0 for i in range(NUM_CLASSES)}
    class_total = {i: 0 for i in range(NUM_CLASSES)}
    
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    
    with torch.no_grad():
        for batch in test_loader:
            if args.phase in ['a', 'b']:
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
            
            test_correct += (preds == y).sum().item()
            test_total += y.size(0)
            
            for p, true_y in zip(preds, y):
                class_correct[true_y.item()] += (p == true_y).item()
                class_total[true_y.item()] += 1
                
    test_acc = test_correct / test_total
    print(f"Test Accuracy: {test_acc:.4f}")
    for i in range(NUM_CLASSES):
        if class_total[i] > 0:
            print(f"  Class {i} ({LABEL_NAMES[i]}): {class_correct[i]/class_total[i]:.4f}")
            
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix
    
    fold_str = f"fold{args.fold}" if args.fold is not None else f"seed{args.seed}"
    
    res_df = pd.DataFrame({'y_true': all_y_true, 'y_pred': all_y_pred})
    for i in range(NUM_CLASSES):
        res_df[f'prob_class_{i}'] = np.array(all_y_prob)[:, i]
    out_csv = os.path.join(args.save_dir, f"predictions_phase{args.phase}_{fold_str}.csv")
    res_df.to_csv(out_csv, index=False)
    
    cm = confusion_matrix(all_y_true, all_y_pred)
    cm_path = os.path.join(args.save_dir, f"cm_phase{args.phase}_{fold_str}.npy")
    np.save(cm_path, cm)
    
    cr = classification_report(all_y_true, all_y_pred, target_names=LABEL_NAMES, output_dict=True, zero_division=0)
    cr_path = os.path.join(args.save_dir, f"classification_report_phase{args.phase}_{fold_str}.json")
    with open(cr_path, 'w', encoding='utf-8') as f:
        json.dump(cr, f, indent=4)
            
    return {
        'test_acc': test_acc,
        'val_acc': best_val_acc,
        'epoch': best_epoch,
        'predictions_path': out_csv,
        'cm_path': cm_path,
        'cr_path': cr_path
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, required=True, choices=['a', 'b', 'f'])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--force_fallback", action="store_true")
    parser.add_argument("--embed_dir", type=str, default="data/embeddings")
    parser.add_argument("--save_dir", type=str, default="runs")
    parser.add_argument("--data_split", type=str, default="data/embeddings/splits.json")
    
    args = parser.parse_args()
    train(args)
