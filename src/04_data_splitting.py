# Chia dữ liệu thành Tập huấn luyện (80%) và Tập kiểm tra (20%). 
# Sử dụng kỹ thuật lấy mẫu phân tầng (stratify=y) để đảm bảo tỷ lệ mất cân bằng của nhãn (loan_status) được giữ nguyên ở cả hai tập dữ liệu.

import pandas as pd
import os
from sklearn.model_selection import train_test_split

def main():
    # Thông tin file
    input_file = "data\\step3_selected_data.csv"
    train_output = "data\\step4_train_data.csv"
    test_output = "data\\step4_test_data.csv"

    # Kiểm tra tồn tại
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} không tồn tại!")
        return

    print("Đang đọc dữ liệu từ Bước 3 (Feature Selection)...")
    df = pd.read_csv(input_file)
    
    if 'loan_status' not in df.columns:
        print("Error: Không tìm thấy cột 'loan_status'!")
        return

    # Tách Features (X) và Target (y)
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']

    # --- BƯỚC 3.3.4: Data Splitting ---
    print(f"\\nTổng số mẫu trước khi chia: {len(df)}")
    print(f"Tỷ lệ mất cân bằng (loan_status) hiện tại:\\n{y.value_counts(normalize=True)}")

    print("\\nĐang tiến hành chia tập dữ liệu (80% Train, 20% Test) với Stratified sampling...")
    
    # Chia dữ liệu dùng stratify=y để giữ nguyên tỷ lệ nhãn
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.20, 
        stratify=y, 
        random_state=42
    )

    # Gộp lại thành DataFrames để lưu
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    print(f"-> Tập Training: {len(train_df)} mẫu (80%)")
    print(f"-> Tỷ lệ nhãn trong tập Train:\\n{train_df['loan_status'].value_counts(normalize=True)}")
    
    print(f"-> Tập Testing: {len(test_df)} mẫu (20%)")
    print(f"-> Tỷ lệ nhãn trong tập Test:\\n{test_df['loan_status'].value_counts(normalize=True)}")

    # Ghi ra file CSV
    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)
    
    print(f"\\nHoàn tất Data Splitting!")
    print(f"Dữ liệu Huấn luyện lưu tại: {train_output}")
    print(f"Dữ liệu Kiểm tra lưu tại: {test_output}")

if __name__ == "__main__":
    main()
