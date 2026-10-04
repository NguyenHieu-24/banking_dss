# Nâng cấp hệ thống — Personalized Credit DSS

Tài liệu này mô tả phần code đã chỉnh sửa/thêm mới để hiện thực hoá **4 cải tiến**
trong `improvement.md`, cùng cách chạy.

## Cấu trúc thư mục sau nâng cấp

```
project/
├── main.py                         # pipeline 6 bước (không đổi)
├── web.py                          # ĐÃ VIẾT LẠI — login + RBAC + tabs + pipeline
├── seed_users.py                   # MỚI — tạo tài khoản khởi tạo (mật khẩu đã băm)
├── requirements.txt                # MỚI
├── .env.example                    # MỚI — mẫu biến môi trường (copy thành .env)
├── data/                           # dữ liệu + model + encoders sinh ra từ pipeline
└── src/
    ├── 01_categorical_encoding.py  # ĐÃ SỬA — lưu encoders.pkl
    ├── 02..06_*.py                 # không đổi
    ├── preprocessing.py            # MỚI — Cải tiến 2 (pipeline suy luận dùng chung)
    ├── security.py                 # MỚI — Cải tiến 4 (.env, băm mật khẩu, che PII)
    ├── database.py                 # MỚI — Cải tiến 1 & 4 (RBAC + cô lập dữ liệu)
    └── auth.py                     # MỚI — Cải tiến 1 (đăng nhập + điều hướng vai trò)
```

## Bốn cải tiến đã thực hiện

**1. Phân quyền Nhân viên / Khách hàng (RBAC).**
`src/auth.py` hiển thị form đăng nhập, kiểm tra mật khẩu băm lấy từ `database`,
rồi `web.py` điều hướng theo vai trò: Khách hàng vào cổng *self-service* (chỉ thấy
kết quả đủ/không đủ điều kiện + gợi ý, **không** thấy SHAP hay xác suất chi tiết);
Nhân viên vào cổng thẩm định đầy đủ.

**2. Tích hợp ML Pipeline cho dữ liệu mới.**
`src/01_categorical_encoding.py` nay lưu lại các `LabelEncoder` thành `encoders.pkl`.
`src/preprocessing.py` dùng lại chính các encoder đó, tái tạo feature engineering
(`debt_to_income_ratio`, `age_to_experience_ratio`) và lọc đúng **9 đặc trưng theo
đúng thứ tự** trước khi đưa vào `model.predict()`.
> Việc này sửa luôn lỗi của bản cũ: `web.py` cũ hardcode `RENT=0` trong khi
> LabelEncoder thực tế cho `RENT=3` (mã hoá theo thứ tự alphabet).

**3. Tách dữ liệu cũ / mới.**
Cổng Nhân viên có các tab riêng: *Historical Dashboard* (đọc `loan_data.csv`, chỉ
hiển thị số liệu tổng hợp/ẩn danh + hiệu suất mô hình), *New Applications* (hồ sơ
khách nộp, lưu ở collection `new_applications`), và *Thẩm định thủ công*.

**4. Bảo mật dữ liệu khách hàng.**
`src/security.py` nạp cấu hình từ `.env` (không hardcode URI/secret), băm mật khẩu
PBKDF2-HMAC-SHA256 (không lưu plaintext), và che PII khi hiển thị (tên, SĐT, CCCD,
email). `src/database.py` ưu tiên MongoDB (thay cho CSV plaintext) với fallback JSON,
và **cô lập dữ liệu ở backend**: khách hàng chỉ truy vấn được hồ sơ của chính mình.

## Cách chạy

```bash
pip install -r requirements.txt

# 1) Huấn luyện pipeline -> sinh encoders.pkl, model .pkl, background data
python main.py

# 2) Cấu hình môi trường (tuỳ chọn MongoDB; bỏ trống MONGO_URI để dùng JSON)
cp .env.example .env        # rồi sửa giá trị

# 3) Tạo tài khoản khởi tạo
python seed_users.py

# 4) Chạy ứng dụng
streamlit run web.py
```

Tài khoản demo: nhân viên `staff01 / Staff@123`, khách hàng `customer01 / Customer@123`
(đổi mật khẩu mặc định qua `.env`).

> **MongoDB:** đặt `MONGO_URI` trong `.env` để dùng Mongo. Nếu để trống, hệ thống
> tự lưu vào `data/db/*.json` để chạy được ngay mà không cần dựng CSDL.
