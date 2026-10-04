# Sử dụng phương pháp chắt lọc 2 tầng (Two-stage selection):
    # Tầng 1: Dùng phương pháp Loại bỏ đệ quy có Kiểm chứng chéo (RFECV) kết hợp với thuật toán Random Forest để loại bỏ các biến không có tác dụng.
    # Tầng 2: Dùng thuật toán Gradient Boosting (SelectFromModel) để chốt lại đúng 9 đặc trưng mạnh nhất.

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.model_selection import StratifiedKFold
import time

def main():
    # Thông tin file
    input_file = "data\\step2_engineered_data.csv"
    output_file = "data\\step3_selected_data.csv"

    # Kiểm tra tồn tại
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} không tồn tại!")
        return

    print("Đang đọc dữ liệu từ Bước 2...")
    df = pd.read_csv(input_file)
    
    # Tách biến mục tiêu (target) và các features (biến độc lập)
    if 'loan_status' not in df.columns:
        print("Error: Không tìm thấy cột 'loan_status' trong dataset!")
        return
        
    y = df['loan_status']
    X = df.drop(columns=['loan_status'])
    
    print(f"Tổng số features ban đầu: {X.shape[1]}")
    print(X.columns.tolist())

    # STAGE 1: Recursive Feature Elimination with Cross-Validation (RFECV)
    print("\\n[STAGE 1] Bắt đầu RFECV để loại bỏ các feature kém hiệu quả...")
    start_time = time.time()
    
    # Sử dụng RandomForest với n_jobs=-1 để tăng tốc độ.
    # Đặt max_depth vừa phải để tránh tốn quá nhiều thời gian fit (45000 dòng x 5 folds)
    estimator_stage1 = RandomForestClassifier(n_estimators=50, max_depth=7, random_state=42, n_jobs=-1)
    
    # Dùng StratifiedKFold để bảo vệ phân phối của y (loan defaults), scoring='f1' vì F1 được quan tâm trong bài
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rfecv = RFECV(estimator=estimator_stage1, step=1, cv=cv, scoring='f1', n_jobs=-1)
    
    rfecv.fit(X, y)
    
    # Lấy các biến sống sót qua Stage 1
    features_stage1 = X.columns[rfecv.support_].tolist()
    X_stage1 = X[features_stage1]
    
    print(f"-> Giai đoạn 1 (RFECV) mất {time.time() - start_time:.2f} giây.")
    print(f"Số features còn lại sau Stage 1: {len(features_stage1)}")
    print(f"Các features được chọn: {features_stage1}")

    # STAGE 2: SelectFromModel with Gradient Boosting / Random Forest
    print("\\n[STAGE 2] Bắt đầu SelectFromModel dựa vào điểm Feature Importance...")
    start_time = time.time()
    
    # Mượn GradientBoosting vì văn bản gợi ý dùng (e.g., Random Forest and Gradient Boosting)
    estimator_stage2 = GradientBoostingClassifier(n_estimators=100, random_state=42)
    
    # SelectFromModel sẽ lấy các features có importance >= threshold mặc định (mean)
    sfm = SelectFromModel(estimator=estimator_stage2, max_features=9, threshold=-np.inf) 
    sfm.fit(X_stage1, y)
    
    # Lấy các biến sống sót qua Stage 2
    features_stage2 = X_stage1.columns[sfm.get_support()].tolist()
    
    print(f"-> Giai đoạn 2 (SelectFromModel) mất {time.time() - start_time:.2f} giây.")
    print(f"Số features CHỐT cuối cùng sau Stage 2: {len(features_stage2)}")
    print(f"Các features mạnh nhất được giữ lại: {features_stage2}")

    # Tạo DataFrame mới chỉ với các features cuối cùng và cột mục tiêu
    final_columns = features_stage2 + ['loan_status']
    df_final = df[final_columns]
    
    # Lưu lại file
    df_final.to_csv(output_file, index=False)
    print(f"\\nHoàn tất Feature Selection! Dữ liệu sau bộ lọc 2 tầng được lưu tại: {output_file}")

if __name__ == "__main__":
    main()
