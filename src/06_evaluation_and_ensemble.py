# Trực quan hóa: Vẽ và lưu các biểu đồ đánh giá như ROC Curve và Feature Importance của từng mô hình.
# Stacking Ensemble: Thay vì dùng Soft Voting cơ bản, hệ thống áp dụng kỹ thuật StackingClassifier phức tạp hơn. 
    # Cụ thể, hệ thống lấy kết quả dự đoán của 4 mô hình (XGBoost, Random Forest, LightGBM, Gradient Boosting) làm dữ liệu đầu vào 
    # để huấn luyện một mô hình thuật toán Meta-learner cuối cùng là Hồi quy Logistic (LogisticRegression).
# Mô hình Ensemble cuối cùng cũng được xuất ra file .pkl để sẵn sàng tích hợp vào giao diện ứng dụng Streamlit.

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

# phần thêm cho StackingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

def main():
    train_file = "data\\step4_train_data.csv"
    test_file = "data\\step4_test_data.csv"

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("Error: Files không tồn tại!")
        return

    print("1. Đang chuẩn bị dữ liệu & Áp dụng SMOTE...")
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    X_train = df_train.drop(columns=['loan_status'])
    y_train = df_train['loan_status']
    X_test = df_test.drop(columns=['loan_status'])
    y_test = df_test['loan_status']
    
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    print("\\n2. Hồi phục 4 Models với Best Hyperparameters (từ Step 5)...")
    xgb = XGBClassifier(objective='binary:logistic', random_state=42, learning_rate=0.1, max_depth=5, n_estimators=100)
    rf = RandomForestClassifier(criterion='gini', random_state=42, max_depth=7, n_estimators=100)
    lgb = LGBMClassifier(objective='binary', random_state=42, verbose=-1, learning_rate=0.1, num_leaves=50)
    gb = GradientBoostingClassifier(loss='log_loss', random_state=42, learning_rate=0.1, max_depth=5, n_estimators=100)

    models = {'XGBoost': xgb, 'Random Forest': rf, 'LightGBM': lgb, 'Gradient Boosting': gb}
    trained_models = {}

    plt.figure(figsize=(10, 8))
    
    print("\\n3. Huấn luyện từng Model & Vẽ ROC Curve...")
    for name, model in models.items():
        model.fit(X_train_sm, y_train_sm)
        trained_models[name] = model
        
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier (AUC = 0.5)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (Section 3.5.4 & Figure 8)')
    plt.legend(loc="lower right")
    plt.savefig('data\\ROC_Curves.png')
    plt.close()
    print("-> Đã lưu biểu đồ ROC tại 'ROC_Curves.png'")

    print("\\n4. Trích xuất Feature Importance...")
    plt.figure(figsize=(12, 8))
    for i, (name, model) in enumerate(trained_models.items(), 1):
        plt.subplot(2, 2, i)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        features = X_test.columns
        sns.barplot(x=importances[indices], y=[features[idx] for idx in indices], orient='h')
        plt.title(f'{name} - Feature Importance')
    plt.tight_layout()
    plt.savefig('data\\Feature_Importance.png')
    plt.close()
    print("-> Đã lưu biểu đồ Feature Importance tại 'Feature_Importance.png'")
    
    # CONTRIBUTION 2
    # Cách tiếp cận Ensemble "ngây thơ"
    # Đoạn code giải quyết: Sử dụng StackingClassifier với mô hình học thuật Meta-learner (LogisticRegression)
    # Cấu trúc:
        # Các mô hình nền tảng (Base Estimators): Hệ thống kết hợp 4 thuật toán mạnh mẽ nhất đã được tinh chỉnh ở các bước trước đó, bao gồm XGBoost, Gradient Boosting, Random Forest và LightGBM.
        # Mô hình tổng hợp (Final Estimator/Meta-learner): Kết quả dự báo từ 4 mô hình trên không được cộng trung bình một cách đơn giản mà được đưa vào một mô hình học máy cấp cao hơn là Hồi quy Logistic (Logistic Regression). 
            # Thuật toán này đóng vai trò "trọng tài", học cách trọng số hóa kết quả của các mô hình nền tảng để đưa ra xác suất nợ xấu cuối cùng chính xác nhất.
    # -> Mô hình Stacking này sau khi huấn luyện thành công được lưu dưới tên file stacking_ensemble_model.pkl. 
        # Đây chính là "bộ não" được ứng dụng Web sử dụng để thực hiện toàn bộ các tính năng từ dự đoán rủi ro, phân tích lợi nhuận cho đến mô phỏng các phương án vay an toàn cho khách hàng.
    print("\n5. Tạo Stacking Ensemble Model (Section 3.6 - Upgrade)...")
    ensemble_clf = StackingClassifier(
        estimators=[
            ('xgb', trained_models['XGBoost']),
            ('gb', trained_models['Gradient Boosting']),
            ('rf', trained_models['Random Forest']),
            ('lgb', trained_models['LightGBM'])
        ],
        final_estimator=LogisticRegression(),
        cv=5,
        n_jobs=-1
    )

    ensemble_clf.fit(X_train_sm, y_train_sm)
    # Lưu mô hình siêu gộp Stacking ra ổ cứng
    joblib.dump(ensemble_clf, 'data\\stacking_ensemble_model.pkl')
    print("\n-> Đã xuất file mô hình Stacking Ensemble cho Streamlit!")
    y_pred_ens = ensemble_clf.predict(X_test)
    y_proba_ens = ensemble_clf.predict_proba(X_test)[:, 1]
    
    fpr_ens, tpr_ens, _ = roc_curve(y_test, y_proba_ens)
    auc_ens = auc(fpr_ens, tpr_ens)
    acc_ens = accuracy_score(y_test, y_pred_ens)
    f1_ens = f1_score(y_test, y_pred_ens)
    
    print("\\n--- Kết quả Đánh giá Ensemble Model ---")
    print(f"Accuracy: {acc_ens:.4f}")
    print(f"F1-Score: {f1_ens:.4f}")
    print(f"ROC AUC : {auc_ens:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_ens))
    print("\\nHoàn thành xử lý Mục 3.5 và 3.6!")

if __name__ == "__main__":
    main()
