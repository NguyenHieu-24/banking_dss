# Tạo thêm 2 features mới có ý nghĩa quan trọng về mặt logic nghiệp vụ:
    # debt_to_income_ratio: Tỷ lệ giữa tổng số nợ và thu nhập của người vay.
    # age_to_experience_ratio: Tỷ lệ giữa tuổi tác và số năm kinh nghiệm.

import pandas as pd
import numpy as np
import os

def main():
    # Thông tin file
    input_file = "data\\step1_encoded_data.csv"
    output_file = "data\\step2_engineered_data.csv"

    # Kiểm tra file tồn tại
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} không tồn tại!")
        return

    print("Đang đọc dữ liệu từ Bước 1...")
    df = pd.read_csv(input_file)

    print("Đang tiến hành Feature Engineering theo mô tả của bài báo...")
    
    # 1. Tạo "debt-to-income ratio" (DTI): tỷ lệ giữa tổng số nợ (loan_amnt) và thu nhập của người vay (person_income)
    df['debt_to_income_ratio'] = df['loan_amnt'] / df['person_income']
    print("- Đã tạo feature: 'debt_to_income_ratio'")

    # 2. Tạo "age-to-experience ratio": tuổi cao nhưng kinh nghiệm = 0 sẽ sinh ra hệ số rủi ro cực lớn
    safe_exp = df['person_emp_exp'].replace(0, 0.5)
    df['age_to_experience_ratio'] = df['person_age'] / safe_exp
    
    # Một cách tiếp cận an toàn khác là thay thế những giá trị infinite bằng phân vị cao nhất (mặc dù việc thêm 0.5 đã tránh được chia cho 0)
    df['age_to_experience_ratio'] = df['age_to_experience_ratio'].replace([np.inf, -np.inf], np.nan)
    if df['age_to_experience_ratio'].isnull().sum() > 0:
        max_ratio = df['age_to_experience_ratio'].max()
        df['age_to_experience_ratio'] = df['age_to_experience_ratio'].fillna(max_ratio)
        
    print("- Đã tạo feature: 'age_to_experience_ratio'")

    df.to_csv(output_file, index=False)
    
    print(f"\\nHoàn tất Feature Engineering! File sau xử lý được lưu tại: {output_file}")
    print("Một vài hàng đầu của dữ liệu mới:")
    print(df[['person_age', 'person_emp_exp', 'age_to_experience_ratio', 'loan_amnt', 'person_income', 'debt_to_income_ratio']].head())

if __name__ == "__main__":
    main()
