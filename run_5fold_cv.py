import argparse
import time
import json
import csv
import numpy as np
from pathlib import Path
from train import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", type=str, nargs='+', default=['a', 'b', 'f'])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--force_fallback", action="store_true")
    parser.add_argument("--output", type=str, default="cv_results.csv")
    
    # We can hardcode or add arguments for directories if needed
    parser.add_argument("--embed_dir", type=str, default="data/embeddings")
    parser.add_argument("--save_dir", type=str, default="runs")
    parser.add_argument("--data_split", type=str, default=None)
    
    args = parser.parse_args()
    
    results = []
    
    start_time = time.time()
    
    print("="*50)
    print("Starting 5-Fold Cross Validation")
    print("="*50)
    
    for phase in args.phases:
        print(f"\nEvaluating Phase {phase.upper()}")
        phase_accs = []
        for fold in range(5):
            print(f"--- Fold {fold} ---")
            
            # Create a namespace for train args
            train_args = argparse.Namespace(
                phase=phase,
                seed=args.seed,
                fold=fold,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                force_fallback=args.force_fallback,
                embed_dir=args.embed_dir,
                save_dir=args.save_dir,
                data_split=args.data_split
            )
            
            res = train(train_args)
            phase_accs.append(res['test_acc'])
            
        mean_acc = np.mean(phase_accs)
        std_acc = np.std(phase_accs)
        
        # Read back predictions to compute overall metrics
        import pandas as pd
        from sklearn.metrics import classification_report, confusion_matrix
        from src.config import LABEL_NAMES
        import os
        
        all_y_true = []
        all_y_pred = []
        sum_cm = None
        
        for fold in range(5):
            csv_path = os.path.join(args.save_dir, f"predictions_phase{phase}_fold{fold}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                all_y_true.extend(df['y_true'].tolist())
                all_y_pred.extend(df['y_pred'].tolist())
                
            cm_path = os.path.join(args.save_dir, f"cm_phase{phase}_fold{fold}.npy")
            if os.path.exists(cm_path):
                cm = np.load(cm_path)
                if sum_cm is None:
                    sum_cm = cm
                else:
                    sum_cm += cm
                    
        f1_macro = 0.0
        if len(all_y_true) > 0:
            cr = classification_report(all_y_true, all_y_pred, target_names=LABEL_NAMES, output_dict=True, zero_division=0)
            f1_macro = cr['macro avg']['f1-score']
            
            summary = {
                'phase': phase,
                'mean_acc': mean_acc,
                'std_acc': std_acc,
                'f1_macro': f1_macro,
                'classification_report': cr,
                'confusion_matrix': sum_cm.tolist() if sum_cm is not None else None
            }
            
            summary_path = os.path.join(args.save_dir, f"summary_phase{phase}.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=4)
            print(f"Đã lưu báo cáo tổng hợp vào {summary_path}")
            print("Classification Report Gộp 5 Folds:")
            print(classification_report(all_y_true, all_y_pred, target_names=LABEL_NAMES, zero_division=0))
        
        results.append({
            'Phase': phase,
            'Fold 0': phase_accs[0],
            'Fold 1': phase_accs[1],
            'Fold 2': phase_accs[2],
            'Fold 3': phase_accs[3],
            'Fold 4': phase_accs[4],
            'Mean': mean_acc,
            'Std': std_acc,
            'F1-Macro': f1_macro
        })
        
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    header = f"{'Phase':<10} | {'Fold 0':<8} | {'Fold 1':<8} | {'Fold 2':<8} | {'Fold 3':<8} | {'Fold 4':<8} | {'Mean':<8} | {'Std':<8} | {'F1-Macro':<8}"
    print(header)
    print("-" * 90)
    for row in results:
        line = f"{row['Phase']:<10} | {row['Fold 0']:.4f}   | {row['Fold 1']:.4f}   | {row['Fold 2']:.4f}   | {row['Fold 3']:.4f}   | {row['Fold 4']:.4f}   | {row['Mean']:.4f}   | {row['Std']:.4f}   | {row['F1-Macro']:.4f}"
        print(line)
        
    print(f"\nTotal time: {total_time/60:.2f} minutes")
    
    # Save to CSV
    if args.output.endswith('.csv'):
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Phase', 'Fold 0', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Mean', 'Std', 'F1-Macro'])
            writer.writeheader()
            writer.writerows(results)
    elif args.output.endswith('.json'):
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=4)
            
    print(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()
