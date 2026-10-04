"""
seed_users.py  —  Tạo tài khoản khởi tạo cho hệ thống (RBAC)

Chạy MỘT LẦN sau khi đã chạy pipeline:
    python seed_users.py

Mật khẩu được băm PBKDF2 trước khi lưu (không lưu plaintext). Có thể đổi mật
khẩu mặc định qua biến môi trường SEED_STAFF_PW / SEED_CUSTOMER_PW.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import database as db
from security import hash_password


DEFAULT_ACCOUNTS = [
    {
        "username": "staff01",
        "name": "Nhân viên Thẩm định",
        "role": db.ROLE_STAFF,
        "email": "underwriter@bank.local",
        "password": os.environ.get("SEED_STAFF_PW", "Staff@123"),
    },
    {
        "username": "customer01",
        "name": "Nguyễn Văn An",
        "role": db.ROLE_CUSTOMER,
        "email": "an.nguyen@example.com",
        "password": os.environ.get("SEED_CUSTOMER_PW", "Customer@123"),
    },
]


def main():
    print(f"Backend tài khoản: {db.backend_name()}")
    for acc in DEFAULT_ACCOUNTS:
        existing = db.get_user(acc["username"])
        if existing:
            print(f"- Bỏ qua (đã tồn tại): {acc['username']}")
            continue
        db.upsert_user(
            username=acc["username"],
            name=acc["name"],
            role=acc["role"],
            password_hash=hash_password(acc["password"]),
            email=acc["email"],
        )
        print(f"- Đã tạo {acc['role']:8s}: {acc['username']}")
    print("\nHoàn tất seed tài khoản.")


if __name__ == "__main__":
    main()
