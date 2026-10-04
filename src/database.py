"""
database.py  —  CẢI TIẾN 1 & 4: Lưu trữ tài khoản (RBAC) + Hồ sơ vay + Cô lập dữ liệu
=====================================================================================

Tầng truy cập dữ liệu thống nhất cho 2 collection:
  * users            : tài khoản đăng nhập + vai trò (staff / customer)
  * new_applications : hồ sơ vay mới nộp từ giao diện

Ưu tiên dùng MongoDB nếu biến môi trường MONGO_URI được cấu hình (Cải tiến 4:
chuyển dữ liệu khỏi CSV plain-text). Nếu không có Mongo (hoặc kết nối lỗi),
TỰ ĐỘNG fallback sang file JSON cục bộ để hệ thống vẫn chạy được ngay.

CÔ LẬP DỮ LIỆU (Data Isolation - Cải tiến 4): mọi truy vấn hồ sơ đều đi qua
list_applications(requester_username, requester_role). Khách hàng CHỈ nhận về
hồ sơ có owner == chính họ; việc lọc thực hiện ở TẦNG BACKEND, không phải chỉ
ẩn trên giao diện.
"""

import os
import json
import uuid
import threading
from datetime import datetime, timezone

from security import get_config

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_DB_DIR = os.path.join(DATA_DIR, "db")
_USERS_JSON = os.path.join(_DB_DIR, "users.json")
_APPS_JSON = os.path.join(_DB_DIR, "new_applications.json")

_LOCK = threading.Lock()

ROLE_STAFF = "staff"
ROLE_CUSTOMER = "customer"


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Chọn backend: Mongo nếu có, ngược lại JSON
# --------------------------------------------------------------------------- #
class _MongoBackend:
    def __init__(self, uri, dbname):
        from pymongo import MongoClient  # import trong hàm để tránh bắt buộc cài
        self._client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        self._client.admin.command("ping")  # ép kiểm tra kết nối ngay
        db = self._client[dbname]
        self.users = db["users"]
        self.apps = db["new_applications"]
        self.users.create_index("username", unique=True)

    def get_user(self, username):
        return self.users.find_one({"username": username}, {"_id": 0})

    def upsert_user(self, doc):
        self.users.update_one({"username": doc["username"]}, {"$set": doc}, upsert=True)

    def list_users(self):
        return list(self.users.find({}, {"_id": 0}))

    def add_application(self, doc):
        self.apps.insert_one(dict(doc))

    def all_applications(self):
        return list(self.apps.find({}, {"_id": 0}))

    def find_applications_by_owner(self, owner):
        return list(self.apps.find({"owner": owner}, {"_id": 0}))

    def update_application_decision(self, app_id, decision, reviewer):
        self.apps.update_one(
            {"app_id": app_id},
            {"$set": {"decision": decision, "reviewed_by": reviewer,
                      "reviewed_at": _now_iso()}},
        )


class _JsonBackend:
    """Fallback cục bộ — đủ dùng để demo/dev khi chưa dựng MongoDB."""

    def __init__(self):
        os.makedirs(_DB_DIR, exist_ok=True)
        for path in (_USERS_JSON, _APPS_JSON):
            if not os.path.exists(path):
                self._write(path, [])

    def _read(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_user(self, username):
        for u in self._read(_USERS_JSON):
            if u.get("username") == username:
                return u
        return None

    def upsert_user(self, doc):
        with _LOCK:
            users = self._read(_USERS_JSON)
            users = [u for u in users if u.get("username") != doc["username"]]
            users.append(doc)
            self._write(_USERS_JSON, users)

    def list_users(self):
        return self._read(_USERS_JSON)

    def add_application(self, doc):
        with _LOCK:
            apps = self._read(_APPS_JSON)
            apps.append(dict(doc))
            self._write(_APPS_JSON, apps)

    def all_applications(self):
        return self._read(_APPS_JSON)

    def find_applications_by_owner(self, owner):
        return [a for a in self._read(_APPS_JSON) if a.get("owner") == owner]

    def update_application_decision(self, app_id, decision, reviewer):
        with _LOCK:
            apps = self._read(_APPS_JSON)
            for a in apps:
                if a.get("app_id") == app_id:
                    a["decision"] = decision
                    a["reviewed_by"] = reviewer
                    a["reviewed_at"] = _now_iso()
            self._write(_APPS_JSON, apps)


_BACKEND = None


def get_backend():
    """Khởi tạo backend 1 lần; tự fallback JSON nếu Mongo không sẵn sàng."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    cfg = get_config()
    uri = cfg["MONGO_URI"].strip()
    if uri:
        try:
            _BACKEND = _MongoBackend(uri, cfg["MONGO_DB"])
            return _BACKEND
        except Exception as e:
            print(f"[database] Không kết nối được MongoDB ({e}); dùng JSON fallback.")
    _BACKEND = _JsonBackend()
    return _BACKEND


def backend_name():
    be = get_backend()
    return "MongoDB" if isinstance(be, _MongoBackend) else "Local JSON"


# --------------------------------------------------------------------------- #
# API cấp cao (web.py / seed_users.py chỉ gọi các hàm này)
# --------------------------------------------------------------------------- #
def get_user(username):
    return get_backend().get_user(username)


def upsert_user(username, name, role, password_hash, email=""):
    if role not in (ROLE_STAFF, ROLE_CUSTOMER):
        raise ValueError(f"Vai trò không hợp lệ: {role}")
    get_backend().upsert_user({
        "username": username,
        "name": name,
        "role": role,
        "email": email,
        "password_hash": password_hash,   # đã băm, KHÔNG bao giờ plaintext
        "created_at": _now_iso(),
    })


def list_users():
    return get_backend().list_users()


def save_application(owner, applicant, raw_features, result):
    """
    Lưu 1 hồ sơ vay mới. `applicant` chứa PII (sẽ được che khi hiển thị),
    `raw_features` là dữ liệu thô khách nhập, `result` là kết quả dự đoán.
    """
    doc = {
        "app_id": uuid.uuid4().hex[:12],
        "owner": owner,                  # username khách hàng -> phục vụ cô lập dữ liệu
        "applicant": applicant,          # {name, phone, national_id, email}
        "features": raw_features,        # giá trị thô (chưa encode)
        "result": result,               # {probability, eligible, ...}
        "decision": "PENDING",
        "created_at": _now_iso(),
    }
    get_backend().add_application(doc)
    return doc["app_id"]


def list_applications(requester_username, requester_role):
    """
    CÔ LẬP DỮ LIỆU ở backend:
      * staff    -> xem được TẤT CẢ hồ sơ (vai trò thẩm định).
      * customer -> CHỈ xem hồ sơ của chính mình (owner == username).
    """
    be = get_backend()
    if requester_role == ROLE_STAFF:
        return be.all_applications()
    return be.find_applications_by_owner(requester_username)


def set_decision(app_id, decision, reviewer):
    get_backend().update_application_decision(app_id, decision, reviewer)
