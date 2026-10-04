"""
auth.py  —  CẢI TIẾN 1: Đăng nhập + Phân quyền (RBAC)
=====================================================

Form đăng nhập cho Streamlit, kiểm tra mật khẩu đã băm (PBKDF2) lấy từ database,
lưu trạng thái vào st.session_state và trả về vai trò để web.py điều hướng giao
diện Khách hàng / Nhân viên.

Lưu ý: bản này tự triển khai gọn nhẹ bằng session_state + băm PBKDF2 (không phụ
thuộc thư viện ngoài để chạy được ngay). Nếu muốn dùng streamlit-authenticator
như gợi ý trong improvement.md, chỉ cần thay phần render form bên dưới — phần
kiểm tra mật khẩu & RBAC vẫn giữ nguyên qua database + security.
"""

import streamlit as st

import database as db
from security import verify_password


def _do_login(username, password):
    user = db.get_user(username)
    if not user:
        return False, "Tài khoản không tồn tại."
    if not verify_password(user.get("password_hash", ""), password):
        return False, "Sai mật khẩu."
    # Chỉ lưu thông tin tối thiểu, KHÔNG lưu mật khẩu vào session
    st.session_state["auth_user"] = {
        "username": user["username"],
        "name": user.get("name", user["username"]),
        "role": user.get("role", db.ROLE_CUSTOMER),
    }
    return True, "OK"


def current_user():
    return st.session_state.get("auth_user")


def logout():
    st.session_state.pop("auth_user", None)


def require_login():
    """
    Hiển thị form login nếu chưa đăng nhập. Trả về dict user nếu đã đăng nhập,
    ngược lại trả None (và web.py nên dừng render phần còn lại).
    """
    user = current_user()
    if user:
        return user

    st.markdown("### 🔐 Đăng nhập hệ thống")
    st.caption(f"Nguồn dữ liệu tài khoản: **{db.backend_name()}**")

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Tên đăng nhập")
        password = st.text_input("Mật khẩu", type="password")
        submitted = st.form_submit_button("Đăng nhập", use_container_width=True)

    if submitted:
        ok, msg = _do_login(username.strip(), password)
        if ok:
            st.rerun()
        else:
            st.error(msg)

    with st.expander("Tài khoản demo (sau khi chạy seed_users.py)"):
        st.markdown(
            "- **Nhân viên:** `staff01` / `Staff@123`\n"
            "- **Khách hàng:** `customer01` / `Customer@123`"
        )
    return None


def render_sidebar_account(user):
    """Hộp thông tin tài khoản + nút đăng xuất ở sidebar."""
    with st.sidebar:
        role_label = "👔 Nhân viên" if user["role"] == db.ROLE_STAFF else "🙋 Khách hàng"
        st.markdown(f"**{user['name']}**")
        st.caption(f"{role_label} · `{user['username']}`")
        if st.button("Đăng xuất", use_container_width=True):
            logout()
            st.rerun()
