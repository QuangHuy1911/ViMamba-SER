import os
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, required=True, choices=['a', 'b', 'f'])
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--save_dir", type=str, default="runs")
    parser.add_argument("--expected_test_samples", type=int, default=1056) # 5280 / 5
    args = parser.parse_args()
    
    print(f"=== Kiểm tra đầu ra cho Phase {args.phase.upper()} - Fold {args.fold} ===")
    
    pred_csv = os.path.join(args.save_dir, f"predictions_phase{args.phase}_fold{args.fold}.csv")
    cm_npy = os.path.join(args.save_dir, f"cm_phase{args.phase}_fold{args.fold}.npy")
    cr_json = os.path.join(args.save_dir, f"classification_report_phase{args.phase}_fold{args.fold}.json")
    
    all_exist = True
    
    if os.path.exists(pred_csv):
        print(f"[OK] Tìm thấy file predictions: {pred_csv}")
        df = pd.read_csv(pred_csv)
        print(f"     -> Số dòng thực tế: {len(df)}")
        if len(df) < args.expected_test_samples * 0.9: # Cho phép du di một chút nếu chia không đều
            print(f"     -> [CẢNH BÁO] Số dòng quá ít so với dự kiến ({args.expected_test_samples})")
    else:
        print(f"[THIẾU] Không tìm thấy: {pred_csv}")
        all_exist = False
        
    if os.path.exists(cm_npy):
        print(f"[OK] Tìm thấy file confusion matrix: {cm_npy}")
    else:
        print(f"[THIẾU] Không tìm thấy: {cm_npy}")
        all_exist = False
        
    if os.path.exists(cr_json):
        print(f"[OK] Tìm thấy file classification report: {cr_json}")
    else:
        print(f"[THIẾU] Không tìm thấy: {cr_json}")
        all_exist = False
        
    print("-" * 50)
    if all_exist:
        print("KẾT LUẬN: Đủ dữ liệu. Có thể tiếp tục chạy fold tiếp theo!")
    else:
        print("KẾT LUẬN: Thiếu dữ liệu. HÃY DỪNG LẠI và kiểm tra log lỗi, không chạy fold tiếp theo!")

if __name__ == "__main__":
    main()
