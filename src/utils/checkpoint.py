import os
import json
import torch
from pathlib import Path
from src.config import LABEL_MAP, LABEL_NAMES

def save_checkpoint(model, optimizer, config_dict, epoch, best_val_acc, save_dir, phase, fold=None, seed=None):
    """
    Lưu checkpoint với metadata đầy đủ.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if fold is not None:
        filename = f"{phase}_fold{fold}_best.pt"
        json_filename = f"{phase}_fold{fold}_config.json"
    else:
        seed_val = seed if seed is not None else 42
        filename = f"{phase}_seed{seed_val}_best.pt"
        json_filename = f"{phase}_seed{seed_val}_config.json"
        
    filepath = os.path.join(save_dir, filename)
    json_filepath = os.path.join(save_dir, json_filename)
    
    # Save config to json
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)
        
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'config': config_dict,
        'label_encoder': {
            'LABEL_MAP': LABEL_MAP,
            'LABEL_NAMES': LABEL_NAMES
        },
        'epoch': epoch,
        'best_val_acc': best_val_acc,
    }
    
    torch.save(checkpoint, filepath)
    return filepath

def load_checkpoint(path, model=None, optimizer=None, device='cpu'):
    """
    Load checkpoint, trả về metadata dict.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")
        
    checkpoint = torch.load(path, map_location=device)
    
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        
    if optimizer is not None and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    return checkpoint
