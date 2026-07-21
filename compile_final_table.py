import argparse
import os
import json
import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import pandas as pd
import numpy as np

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_csv(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def extract_mean_std(data_dict, phase_name):
    # Dùng cho các file dictionary đơn giản
    if not data_dict:
        return "N/A"
    
    # Tìm key giống phase_name
    for k, v in data_dict.items():
        if phase_name.lower() in k.lower():
            if isinstance(v, (int, float)):
                return f"{v:.2f}"
    
    # Nếu file là list (như file json từ run_5fold_cv.py)
    if isinstance(data_dict, list):
        for item in data_dict:
            if item.get("Phase", "").lower() == phase_name.lower():
                mean = item.get("Mean", 0) * 100
                std = item.get("Std", 0) * 100
                return f"{mean:.2f} ± {std:.2f}"
                
    return "N/A"

def extract_from_cv_df(df, phase_name):
    if df is None:
        return "N/A"
    row = df[df['Phase'].str.lower() == phase_name.lower()]
    if not row.empty:
        mean = row['Mean'].values[0] * 100
        std = row['Std'].values[0] * 100
        return f"{mean:.2f} ± {std:.2f}"
    return "N/A"

def main(args):
    print("--- Tổng hợp bảng kết quả cuối kỳ ---")
    
    results = []
    
    # 1. Phase A
    # Thử lấy từ CV file trước
    cv_df = load_csv(args.cv_results)
    cv_json = load_json(args.cv_results) if args.cv_results.endswith('.json') else None
    
    phase_a_val = extract_from_cv_df(cv_df, 'a') if cv_df is not None else extract_mean_std(cv_json, 'a')
    
    if phase_a_val == "N/A":
        # Fallback to midterm file
        midterm_a = load_json(args.phase_a)
        phase_a_val = extract_mean_std(midterm_a, "A1")
        if phase_a_val == "N/A":
            print(f"CẢNH BÁO: Không tìm thấy kết quả Phase A trong {args.cv_results} hay {args.phase_a}")
            
    # 2. Phase B (B3b)
    phase_b_val = extract_from_cv_df(cv_df, 'b') if cv_df is not None else extract_mean_std(cv_json, 'b')
    if phase_b_val == "N/A":
        midterm_b = load_json(args.phase_b)
        phase_b_val = extract_mean_std(midterm_b, "B3b")
        if phase_b_val == "N/A":
            phase_b_val = extract_mean_std(midterm_b, "B2") # Fallback to B2 if B3b not found
        if phase_b_val == "N/A":
            print(f"CẢNH BÁO: Không tìm thấy kết quả Phase B trong {args.cv_results} hay {args.phase_b}")

    # 3. Phase F
    phase_f_val = extract_from_cv_df(cv_df, 'f') if cv_df is not None else extract_mean_std(cv_json, 'f')
    if phase_f_val == "N/A":
        phase_f_data = load_json(args.phase_f)
        if phase_f_data and 'test_acc' in phase_f_data:
            phase_f_val = f"{phase_f_data['test_acc']*100:.2f}"
        else:
            print(f"CẢNH BÁO: Không tìm thấy kết quả Phase F trong {args.cv_results} hay {args.phase_f}")

    # Format table
    df_results = pd.DataFrame([
        {"Phase": "A1 (Audio-only MLP)", "Accuracy (%)": phase_a_val},
        {"Phase": "B3b (Weighted Fusion)", "Accuracy (%)": phase_b_val},
        {"Phase": "F (TME + Bi-Sequence Fusion)", "Accuracy (%)": phase_f_val}
    ])
    
    # Save CSV
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "final_results_table.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nĐã lưu bảng CSV tại: {csv_path}")
    
    # Save Markdown
    md_path = os.path.join(args.out_dir, "final_results_table.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Bảng kết quả tổng hợp\n\n")
        f.write(df_results.to_markdown(index=False))
        
    print(f"Đã lưu bảng Markdown tại: {md_path}\n")
    
    print("BẢNG KẾT QUẢ:")
    print(df_results.to_markdown(index=False))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase_a", type=str, default="runs/midterm_results/phase_B_results.json", help="File json chứa kết quả A1")
    parser.add_argument("--phase_b", type=str, default="runs/midterm_results/phase_B_results.json", help="File json chứa kết quả B3b/B2")
    parser.add_argument("--cv_results", type=str, default="cv_results.csv", help="File CSV/JSON từ run_5fold_cv.py")
    parser.add_argument("--phase_f", type=str, default="runs/phase_f_results.json", help="File JSON kết quả riêng của Phase F")
    parser.add_argument("--out_dir", type=str, default="results")
    
    args = parser.parse_args()
    main(args)
