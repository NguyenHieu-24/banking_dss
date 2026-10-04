"""
security.py  —  CẢI TIẾN 4: Bảo mật Dữ liệu Khách hàng
======================================================

Gom toàn bộ tiện ích bảo mật vào một chỗ:

  * Cấu hình qua biến môi trường (.env) — KHÔNG hardcode URI/secret trong code.
  * Băm mật khẩu an toàn (PBKDF2-HMAC-SHA256, có salt) — KHÔNG lưu plaintext.
  * Ẩn danh / che dữ liệu PII (tên, SĐT, CCCD, email) khi hiển thị.

Mật khẩu không bao giờ được lưu hay log ở dạng rõ. Hàm verify dùng so sánh
hằng-thời-gian (hmac.compare_digest) để chống tấn công đo thời gian.
"""

import os
import base64
import hashlib
import hmac
import secrets

# Nạp .env nếu có python-dotenv (Cải tiến 4: dùng Environment Variables)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # python-dotenv chưa cài -> vẫn đọc được os.environ bình thường
    pass


# --------------------------------------------------------------------------- #
# Cấu hình lấy từ môi trường (có giá trị mặc định an toàn cho lúc dev)
# --------------------------------------------------------------------------- #
def get_config():
    return {
        "MONGO_URI": os.environ.get("MONGO_URI", ""),           # rỗng -> fallback JSON
        "MONGO_DB": os.environ.get("MONGO_DB", "credit_dss"),
        "APP_SECRET": os.environ.get("APP_SECRET", "dev-only-insecure-secret-change-me"),
        "PBKDF2_ITERATIONS": int(os.environ.get("PBKDF2_ITERATIONS", "200000")),
    }


# --------------------------------------------------------------------------- #
# Băm & kiểm tra mật khẩu (PBKDF2-HMAC-SHA256)
# Định dạng lưu: pbkdf2_sha256$<iterations>$<salt_b64>$<hash_b64>
# --------------------------------------------------------------------------- #
def hash_password(password: str, iterations: int | None = None) -> str:
    if iterations is None:
        iterations = get_config()["PBKDF2_ITERATIONS"]
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "pbkdf2_sha256${}${}${}".format(
        iterations,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(dk).decode("ascii"),
    )


def verify_password(stored: str, password: str) -> bool:
    try:
        algo, iter_s, salt_b64, hash_b64 = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        iterations = int(iter_s)
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Ẩn danh / che dữ liệu PII (Data Masking)
# --------------------------------------------------------------------------- #
def mask_name(name: str) -> str:
    """'Nguyễn Văn An' -> 'N*** V*** A***' (giữ chữ cái đầu mỗi từ)."""
    if not name:
        return ""
    parts = [p for p in str(name).split() if p]
    return " ".join(p[0] + "*" * max(len(p) - 1, 1) for p in parts)


def mask_phone(phone: str) -> str:
    """'0901234567' -> '090****567' (giữ 3 đầu + 3 cuối)."""
    s = "".join(ch for ch in str(phone) if ch.isdigit())
    if len(s) <= 6:
        return "*" * len(s)
    return s[:3] + "*" * (len(s) - 6) + s[-3:]


def mask_national_id(nid: str) -> str:
    """'079201001234' -> '0792*****234' (giữ 4 đầu + 3 cuối)."""
    s = str(nid)
    if len(s) <= 7:
        return "*" * len(s)
    return s[:4] + "*" * (len(s) - 7) + s[-3:]


def mask_email(email: str) -> str:
    """'an.nguyen@bank.com' -> 'a***@bank.com'."""
    if not email or "@" not in str(email):
        return "***"
    local, domain = str(email).split("@", 1)
    head = local[0] if local else "*"
    return f"{head}{'*' * max(len(local) - 1, 1)}@{domain}"
