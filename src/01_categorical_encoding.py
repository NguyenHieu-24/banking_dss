# Tự động phát hiện các cột chứa dữ liệu dạng chữ (object) và sử dụng thuật toán
# LabelEncoder của scikit-learn để chuyển đổi chúng sang dạng số.
#
# === CẢI TIẾN 2 (ML Pipeline Integration) ===
# Thay vì dùng CHUNG một LabelEncoder rồi vứt đi, ta tạo MỘT encoder RIÊNG cho
# từng cột và LƯU LẠI toàn bộ bằng joblib. Nhờ vậy giao diện Streamlit (web.py)
# có thể nạp lại đúng các encoder này để mã hoá dữ liệu thô y hệt lúc huấn luyện,
# tránh tình trạng "hardcode mapping sai" như bản cũ.

import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Thư mục data nằm cạnh thư mục src (tính theo vị trí file, không phụ thuộc CWD)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main():
    input_file = os.path.join(DATA_DIR, "loan_data.csv")
    output_file = os.path.join(DATA_DIR, "step1_encoded_data.csv")
    encoders_file = os.path.join(DATA_DIR, "encoders.pkl")  # <-- MỚI

    if not os.path.exists(input_file):
        print(f"Error: File {input_file} không tồn tại!")
        return

    print("Đang đọc dữ liệu...")
    df = pd.read_csv(input_file)

    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
    print(f"Các cột phân loại sẽ Label Encoding: {categorical_columns}")

    # MỖI cột một encoder riêng -> lưu vào dict để tái sử dụng khi suy luận
    encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        mapping = dict(zip(le.classes_, range(len(le.classes_))))
        print(f"Mapping cho cột '{col}': {mapping}")

    if "loan_status" in df.columns and "loan_status" not in categorical_columns:
        le_target = LabelEncoder()
        df["loan_status"] = le_target.fit_transform(df["loan_status"].astype(str))
        encoders["loan_status"] = le_target
        print(f"Mapping bổ sung 'loan_status': "
              f"{dict(zip(le_target.classes_, range(len(le_target.classes_))))}")

    df.to_csv(output_file, index=False)
    print(f"\nHoàn tất! Dữ liệu mã hoá lưu tại: {output_file}")

    # === ĐIỂM MẤU CHỐT CẢI TIẾN 2: lưu lại bộ encoders ===
    joblib.dump(encoders, encoders_file)
    print(f"-> Đã lưu bộ LabelEncoders tại: {encoders_file}")
    print(f"   (web.py sẽ nạp lại file này để mã hoá dữ liệu khách hàng nhập vào)")


if __name__ == "__main__":
    main()
