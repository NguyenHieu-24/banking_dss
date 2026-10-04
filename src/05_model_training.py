# Xử lý mất cân bằng: Áp dụng kỹ thuật sinh mẫu nhân tạo SMOTE (imblearn) trên tập Train để cân bằng số lượng giữa các lớp.
# Huấn luyện 4 thuật toán: Tiến hành train 4 thuật toán mạnh mẽ: XGBoost, Random Forest, LightGBM, và Gradient Boosting.
# Tối ưu hóa (Bayesian Optimization): Sử dụng thư viện optuna để tự động dò tìm bộ siêu tham số tốt nhất (Best Hyperparameters) cho từng mô hình thông qua Cross-Validation 
    # thay vì dùng GridSearch truyền thống.
# Lưu các model tốt nhất (.pkl) và trích xuất thêm tập dữ liệu nền (background data) phục vụ cho kỹ thuật giải thích mô hình SHAP về sau.

import joblib
import pandas as pd
import numpy as np
import os
import time
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier

# phần thêm cho Bayesian Optimization
import optuna
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')

def evaluate_model(model_name, model, X_test, y_test, train_time):
    """Tiện ích đánh giá và in ra các chỉ số"""
    start_test = time.time()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    test_time = time.time() - start_test
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    print(f"--- Kết quả cho {model_name} ---")
    print(f"Training Time: {train_time:.2f} s | Testing Time: {test_time:.2f} s")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("-" * 35)
    
    return [model_name, acc, f1, roc_auc, prec, rec, train_time]

def main():
    train_file = "data\\step4_train_data.csv"
    test_file = "data\\step4_test_data.csv"

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("Error: Không tìm thấy dữ liệu Train/Test từ Bước 4!")
        return

    print("1. Đang tải dữ liệu...")
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    X_train = df_train.drop(columns=['loan_status'])
    y_train = df_train['loan_status']
    X_test = df_test.drop(columns=['loan_status'])
    y_test = df_test['loan_status']

    # 3.4.5. Addressing Class Imbalance (SMOTE)
    print(f"\\n2. Xử lý mất cân bằng lớp (SMOTE)...")
    print(f"Tỷ lệ trước khi SMOTE:\\n{y_train.value_counts()}")
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"Tỷ lệ sau khi SMOTE:\\n{y_train_sm.value_counts()}")
    
    # Tính scale_pos_weight cho mục 3.4.5 (Class weighting + SMOTE dual approach)
    # Tuy nhiên, do SMOTE đã làm cân bằng tỷ lệ (1:1), 'balanced' class weight sẽ là 1:1 mặc định.
    # Trong code, ta khai báo cẩn thận theo lý thuyết class weight nếu mô hình cho phép.
    
    # 3.4.6 & 3.4.7. Hyperparameter Tuning & Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    
    # 1. XGBoost
    print("\n[1/4] Đang huấn luyện XGBoost (Tối ưu bằng Optuna)...")
    start = time.time()

    # CONTRIBUTION 1
    # Thiếu vắng việc tối ưu hóa siêu tham số (Hyperparameter Tuning)
    # Đoạn code giải quyết: Khởi tạo hàm objective và dùng Bayesian Optimization để dò tìm
    def objective_xgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
        }
        
        # Khởi tạo mô hình thử nghiệm
        model = XGBClassifier(**params, random_state=42, objective='binary:logistic', eval_metric='logloss', n_jobs=-1)
        
        # Đánh giá bằng Cross-validation (Sử dụng biến cv đã khai báo ở trên)
        score = cross_val_score(model, X_train_sm, y_train_sm, cv=cv, scoring='f1', n_jobs=-1).mean()
        return score

    # Khởi chạy Optuna (Tắt bớt log để màn hình terminal đỡ rối)
    optuna.logging.set_verbosity(optuna.logging.WARNING) 
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(objective_xgb, n_trials=20) 
        
    print(f"Best Params XGBoost: {study_xgb.best_params}")

    # Train lại mô hình CHÍNH THỨC bằng bộ tham số tốt nhất
    best_xgb = XGBClassifier(**study_xgb.best_params, random_state=42, objective='binary:logistic', eval_metric='logloss', n_jobs=-1)
    best_xgb.fit(X_train_sm, y_train_sm)
    
    print("\nĐang xuất file Model và Background Data cho Streamlit...")
    # 1. Lưu mô hình XGBoost
    joblib.dump(best_xgb, 'data\\xgboost_loan_model.pkl')
    
    # 2. Lưu 100 mẫu ngẫu nhiên làm dữ liệu nền (background data) cho SHAP chạy mượt
    background_data = X_train_sm.sample(n=100, random_state=42)
    background_data.to_csv('data\\shap_background_data.csv', index=False)
    print("-> Đã lưu 'xgboost_loan_model.pkl' và 'shap_background_data.csv' thành công!")
    
    train_time = time.time() - start
    
    # Đưa vào hàm đánh giá hệ thống
    res_xgb = evaluate_model("XGBoost", best_xgb, X_test, y_test, train_time)
    results.append(res_xgb)
    
    # 2. Random Forest
    print("\n[2/4] Đang huấn luyện Random Forest (Tối ưu bằng Optuna)...")
    start = time.time()

    def objective_rf(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        }
        
        model = RandomForestClassifier(**params, criterion='gini', class_weight='balanced', random_state=42, n_jobs=-1)
        score = cross_val_score(model, X_train_sm, y_train_sm, cv=cv, scoring='f1', n_jobs=-1).mean()
        return score

    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(objective_rf, n_trials=20) 
    
    print(f"Best Params Random Forest: {study_rf.best_params}")

    # Train lại mô hình CHÍNH THỨC
    best_rf = RandomForestClassifier(**study_rf.best_params, criterion='gini', class_weight='balanced', random_state=42, n_jobs=-1)
    best_rf.fit(X_train_sm, y_train_sm)

    print("\nĐang xuất file Model Random Forest...")
    joblib.dump(best_rf, 'data\\rf_loan_model.pkl')
    print("-> Đã lưu 'rf_loan_model.pkl' thành công!")
    
    train_time = time.time() - start
    res_rf = evaluate_model("Random Forest", best_rf, X_test, y_test, train_time)
    results.append(res_rf)

    # 3. LightGBM
    print("\n[3/4] Đang huấn luyện LightGBM (Tối ưu bằng Optuna)...")
    start = time.time()

    def objective_lgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 60),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 40)
        }
        
        model = LGBMClassifier(**params, objective='binary', class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1)
        score = cross_val_score(model, X_train_sm, y_train_sm, cv=cv, scoring='f1', n_jobs=-1).mean()
        return score

    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(objective_lgb, n_trials=20) 
    
    print(f"Best Params LightGBM: {study_lgb.best_params}")

    # Train lại mô hình CHÍNH THỨC
    best_lgb = LGBMClassifier(**study_lgb.best_params, objective='binary', class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1)
    best_lgb.fit(X_train_sm, y_train_sm)
    
    print("\nĐang xuất file Model LightGBM...")
    joblib.dump(best_lgb, 'data\\lgb_loan_model.pkl')
    print("-> Đã lưu 'lgb_loan_model.pkl' thành công!")
    
    train_time = time.time() - start
    res_lgb = evaluate_model("LightGBM", best_lgb, X_test, y_test, train_time)
    results.append(res_lgb)

    # 4. Gradient Boosting
    print("\n[4/4] Đang huấn luyện Gradient Boosting (Tối ưu bằng Optuna)...")
    start = time.time()

    def objective_gb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
        }
        
        model = GradientBoostingClassifier(**params, loss='log_loss', random_state=42)
        score = cross_val_score(model, X_train_sm, y_train_sm, cv=cv, scoring='f1', n_jobs=-1).mean()
        return score

    study_gb = optuna.create_study(direction='maximize')
    study_gb.optimize(objective_gb, n_trials=20) 
    
    print(f"Best Params Gradient Boosting: {study_gb.best_params}")

    # Train lại mô hình CHÍNH THỨC
    best_gb = GradientBoostingClassifier(**study_gb.best_params, loss='log_loss', random_state=42)
    best_gb.fit(X_train_sm, y_train_sm)
    
    print("\nĐang xuất file Model Gradient Boosting...")
    joblib.dump(best_gb, 'data\\gb_loan_model.pkl')
    print("-> Đã lưu 'gb_loan_model.pkl' thành công!")
    
    train_time = time.time() - start
    res_gb = evaluate_model("Gradient Boosting", best_gb, X_test, y_test, train_time)
    results.append(res_gb)

    # Tổng kết Output
    df_results = pd.DataFrame(results, columns=['Model', 'Accuracy', 'F1-Score', 'ROC AUC', 'Precision', 'Recall', 'Train_Time(s)'])
    print("\n=======================================================")
    print("BẢNG TỔNG HỢP KẾT QUẢ CÁC MÔ HÌNH TRÊN TẬP TESTING")
    print("=======================================================")
    print(df_results.to_string(index=False))
    
    df_results.to_csv("data\\step5_model_results.csv", index=False)
    print("\n-> Toàn bộ kết quả đã được lưu vào 'step5_model_results.csv'")

if __name__ == "__main__":
    main()
