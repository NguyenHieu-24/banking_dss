"""
preprocessing.py  —  CẢI TIẾN 2: ML Pipeline Integration
========================================================

Tái tạo CHÍNH XÁC pipeline tiền xử lý đã dùng khi huấn luyện, để mọi dòng dữ
liệu THÔ nhập từ giao diện Streamlit đều đi qua đúng 3 bước trước khi vào model:

    1) Categorical Encoding  -> dùng lại các LabelEncoder đã lưu ở Bước 1
    2) Feature Engineering   -> tái tạo logic của Bước 2 (DTI, age/experience)
    3) Feature Selection     -> giữ ĐÚNG 9 đặc trưng & ĐÚNG thứ tự cột của Bước 3

Nhờ module dùng chung này, web.py không còn phải "đoán" mapping (bản cũ hardcode
home_map sai: RENT=0 trong khi LabelEncoder thực tế cho RENT=3).
"""

import os
import joblib
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ENCODERS_PATH = os.path.join(DATA_DIR, "encoders.pkl")
BACKGROUND_PATH = os.path.join(DATA_DIR, "shap_background_data.csv")

# Thứ tự 9 đặc trưng chốt ở Bước 3 (dùng làm fallback nếu thiếu background file).
# Ở runtime ta ưu tiên đọc header của shap_background_data.csv để luôn khớp pipeline.
_FALLBACK_SELECTED_FEATURES = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "loan_amnt",
    "loan_intent",
    "loan_int_rate",
    "credit_score",
    "previous_loan_defaults_on_file",
    "debt_to_income_ratio",
]

# Các cột phân loại có thể xuất hiện trong input thô (key = tên cột).
_CATEGORICAL_COLS = [
    "person_gender",
    "person_education",
    "person_home_ownership",
    "loan_intent",
    "previous_loan_defaults_on_file",
]


# Nạp tài nguyên (cache đơn giản ở cấp module)
_ENCODERS_CACHE = None
_BACKGROUND_CACHE = None


def load_encoders():
    """Nạp dict {tên_cột: LabelEncoder} đã lưu ở Bước 1."""
    global _ENCODERS_CACHE
    if _ENCODERS_CACHE is None:
        if not os.path.exists(ENCODERS_PATH):
            raise FileNotFoundError(
                f"Không tìm thấy {ENCODERS_PATH}. Hãy chạy lại Bước 1 "
                f"(01_categorical_encoding.py) để sinh encoders.pkl."
            )
        _ENCODERS_CACHE = joblib.load(ENCODERS_PATH)
    return _ENCODERS_CACHE


def load_background():
    """Nạp dữ liệu nền SHAP (đồng thời là nguồn chuẩn về tên + thứ tự 9 features)."""
    global _BACKGROUND_CACHE
    if _BACKGROUND_CACHE is None and os.path.exists(BACKGROUND_PATH):
        _BACKGROUND_CACHE = pd.read_csv(BACKGROUND_PATH)
    return _BACKGROUND_CACHE


def get_selected_features():
    """Trả về danh sách 9 features theo đúng thứ tự pipeline đã chốt."""
    bg = load_background()
    if bg is not None and len(bg.columns) > 0:
        return bg.columns.tolist()
    return list(_FALLBACK_SELECTED_FEATURES)


def get_category_options(col):
    """Lấy danh sách nhãn gốc (vd ['MORTGAGE','OTHER','OWN','RENT']) cho dropdown UI."""
    enc = load_encoders().get(col)
    if enc is None:
        return []
    return list(enc.classes_)


# Bước 2 (tái tạo): Feature Engineering
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo lại các đặc trưng phái sinh y hệt 02_feature_engineering.py."""
    df = df.copy()

    if "loan_amnt" in df.columns and "person_income" in df.columns:
        df["debt_to_income_ratio"] = np.where(
            df["person_income"] > 0,
            df["loan_amnt"] / df["person_income"],
            0.0,
        )

    if "person_age" in df.columns and "person_emp_exp" in df.columns:
        safe_exp = df["person_emp_exp"].replace(0, 0.5)
        ratio = df["person_age"] / safe_exp
        ratio = ratio.replace([np.inf, -np.inf], np.nan)
        df["age_to_experience_ratio"] = ratio.fillna(ratio.max())

    return df


# Hàm tiền xử lý chính cho 1 hồ sơ thô từ Streamlit
def preprocess_raw_input(raw: dict) -> pd.DataFrame:
    """
    Nhận 1 dict dữ liệu THÔ (giá trị phân loại để ở dạng CHỮ gốc, ví dụ
    'RENT', 'PERSONAL', 'No') và trả về DataFrame 1 dòng đã sẵn sàng cho
    model.predict(): đã encode đúng, đã tạo feature phái sinh, đã lọc & sắp
    đúng 9 cột theo thứ tự pipeline.
    """
    encoders = load_encoders()
    df = pd.DataFrame([dict(raw)])

    # (1) Encoding các cột phân loại bằng encoder đã lưu
    for col in _CATEGORICAL_COLS:
        if col in df.columns and col in encoders:
            enc = encoders[col]
            val = str(df.at[0, col])
            if val in set(enc.classes_):
                df[col] = int(enc.transform([val])[0])
            else:
                # Nhãn lạ chưa từng thấy -> gán lớp đầu tiên (an toàn, tránh crash)
                df[col] = 0

    # (2) Feature Engineering
    df = engineer_features(df)

    # (3) Feature Selection: giữ đúng 9 cột & đúng thứ tự
    selected = get_selected_features()
    bg = load_background()
    for col in selected:
        if col not in df.columns:
            # Điền median từ background nếu thiếu (an toàn theo phân phối train)
            df[col] = float(bg[col].median()) if bg is not None and col in bg.columns else 0.0

    df = df[selected].astype(float)
    return df


def compute_dti(loan_amnt: float, person_income: float) -> float:
    """Tiện ích tính nhanh DTI cho phần hiển thị/đề xuất."""
    return (loan_amnt / person_income) if person_income and person_income > 0 else 0.0
