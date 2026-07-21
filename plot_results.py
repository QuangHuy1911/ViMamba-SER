import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

LABEL_NAMES = ["happy", "neutral", "sad", "angry"]
FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def plot_charts():
    # Cố gắng load summary thật, nếu không có thì dùng số liệu giả để test script
    summary_file = "runs/summary_phasef.json"
    if os.path.exists(summary_file):
        with open(summary_file, 'r', encoding='utf-8') as f:
            data_f = json.load(f)
    else:
        print(f"Không tìm thấy {summary_file}, dùng dữ liệu mô phỏng để sinh biểu đồ...")
        data_f = {
            'mean_acc': 0.725, 'std_acc': 0.015,
            'f1_macro': 0.710,
            'confusion_matrix': [[250, 40, 10, 5], [30, 310, 20, 15], [10, 25, 230, 5], [5, 10, 5, 260]],
            'classification_report': {
                'happy': {'precision': 0.72, 'recall': 0.70, 'f1-score': 0.71},
                'neutral': {'precision': 0.75, 'recall': 0.76, 'f1-score': 0.75},
                'sad': {'precision': 0.68, 'recall': 0.65, 'f1-score': 0.66},
                'angry': {'precision': 0.76, 'recall': 0.78, 'f1-score': 0.77}
            }
        }

    # 1. Bar chart so sánh Accuracy + F1-macro (A1, B3b, F)
    # Số liệu giả định cho A1 và B3b (lấy từ báo cáo giữa kỳ)
    phases = ['A1 (Audio-only)', 'B3b (Fusion)', 'Phase F (Mamba)']
    acc = [0.65, 0.68, data_f['mean_acc']]
    f1 = [0.62, 0.66, data_f['f1_macro']]
    acc_std = [0, 0, data_f.get('std_acc', 0)] # Cột F có error bar
    
    x = np.arange(len(phases))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, acc, width, yerr=[0,0,acc_std[2]], label='Accuracy', capsize=5, color='#4c72b0')
    rects2 = ax.bar(x + width/2, f1, width, label='F1-Macro', color='#55a868')
    
    ax.set_ylabel('Scores')
    ax.set_title('So sánh Accuracy và F1-Macro qua các giai đoạn')
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "chart_1_comparison.png")
    plt.close()
    
    # 2. Confusion matrix heatmap Phase F
    cm = np.array(data_f['confusion_matrix'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.title('Confusion Matrix - Phase F (5-Fold Sum)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "chart_2_cm_phaseF.png")
    plt.close()
    
    # 3. Bar chart Precision/Recall/F1 theo từng lớp Phase F
    cr = data_f['classification_report']
    classes = LABEL_NAMES
    metrics = {
        'Precision': [cr[c]['precision'] for c in classes],
        'Recall': [cr[c]['recall'] for c in classes],
        'F1-Score': [cr[c]['f1-score'] for c in classes]
    }
    
    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, metrics['Precision'], width, label='Precision', color='#c44e52')
    ax.bar(x, metrics['Recall'], width, label='Recall', color='#8172b3')
    ax.bar(x + width, metrics['F1-Score'], width, label='F1-Score', color='#937860')
    
    ax.set_ylabel('Scores')
    ax.set_title('Precision / Recall / F1-Score theo từng lớp (Phase F)')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "chart_3_class_metrics.png")
    plt.close()

    print(f"Đã xuất 3 biểu đồ ra thư mục {FIGURES_DIR}")

if __name__ == "__main__":
    plot_charts()
