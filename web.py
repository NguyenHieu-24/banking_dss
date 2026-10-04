"""
web.py  —  Personalized Credit Decision Support System
======================================================
Bản nâng cấp tích hợp đầy đủ 4 cải tiến:

  1) RBAC: đăng nhập + phân quyền Nhân viên / Khách hàng (src/auth.py).
  2) ML Pipeline Integration: dữ liệu thô đi qua src/preprocessing.py (encoder đã
     lưu + feature engineering + đúng 9 features) trước khi vào model.
  3) Tách dữ liệu cũ/mới: giao diện Nhân viên có tab "Historical Dashboard" và tab
     "New Applications" riêng biệt.
  4) Bảo mật: .env cho cấu hình, che PII khi hiển thị, cô lập dữ liệu ở backend.

Chạy:  streamlit run web.py
(nhớ đã chạy pipeline để có model + encoders, và chạy seed_users.py để có tài khoản)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import auth
import database as db
import preprocessing as pp
from security import mask_name, mask_phone, mask_national_id, mask_email

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_PATH = os.path.join(DATA_DIR, "stacking_ensemble_model.pkl")
RAW_DATA_PATH = os.path.join(DATA_DIR, "loan_data.csv")
RESULTS_PATH = os.path.join(DATA_DIR, "step5_model_results.csv")

st.set_page_config(page_title="Personalized Credit Risk DSS", page_icon="🏦", layout="wide")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_background():
    return pp.load_background()


@st.cache_resource
def get_shap_explainer(_model, _bg):
    import shap

    def predict_class_1(X):
        return _model.predict_proba(X)[:, 1]

    return shap.KernelExplainer(predict_class_1, _bg.sample(n=50, random_state=42))


def predict_proba(model, raw: dict) -> float:
    X = pp.preprocess_raw_input(raw)
    return float(model.predict_proba(X)[0][1])


def find_safe_amount(model, raw: dict, start_amt: int, floor: int = 500, step: int = 100):
    """Quét lùi tìm hạn mức vay cao nhất có xác suất vỡ nợ < 50%."""
    for amt in range(int(start_amt), floor - 1, -step):
        trial = dict(raw)
        trial["loan_amnt"] = amt
        if predict_proba(model, trial) < 0.5:
            return amt, predict_proba(model, trial)
    return None, None


def loan_input_form(key_prefix=""):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        person_age = st.number_input("Tuổi", 18, 100, 30, key=key_prefix + "age")
        person_income = st.number_input("Thu nhập năm ($)", 10000, 500000, 50000,
                                         step=1000, key=key_prefix + "income")
    with col2:
        home_opts = pp.get_category_options("person_home_ownership") or \
            ["MORTGAGE", "OTHER", "OWN", "RENT"]
        person_home_ownership = st.selectbox("Sở hữu nhà ở", home_opts, key=key_prefix + "home")
        default_choice = st.radio("Có nợ xấu trước đây?", ["No", "Yes"],
                                  horizontal=True, key=key_prefix + "def")
    with col3:
        loan_amnt = st.number_input("Số tiền vay ($)", 500, 100000, 10000,
                                    step=500, key=key_prefix + "amt")
        intent_opts = pp.get_category_options("loan_intent") or \
            ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"]
        loan_intent = st.selectbox("Mục đích vay", intent_opts, key=key_prefix + "intent")
    with col4:
        loan_int_rate = st.number_input("Lãi suất (%)", 5.0, 35.0, 10.5,
                                        step=0.1, key=key_prefix + "rate")
        credit_score = st.number_input("Điểm tín dụng", 300, 850, 650,
                                       step=1, key=key_prefix + "score")

    dti = pp.compute_dti(loan_amnt, person_income)
    st.info(f"**Tỷ lệ Nợ/Thu nhập (DTI):** {dti:.2f}")

    raw = {
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "loan_amnt": loan_amnt,
        "loan_intent": loan_intent,
        "loan_int_rate": loan_int_rate,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": default_choice,
    }
    return raw, loan_amnt, loan_int_rate


def staff_full_analysis(model, bg, raw, loan_amnt, int_rate):
    X = pp.preprocess_raw_input(raw)
    proba = float(model.predict_proba(X)[0][1])

    c1, c2 = st.columns(2)
    with c1:
        st.write("### 🎯 Kết quả dự đoán")
        if proba > 0.5:
            st.error(f"**CẢNH BÁO: TỪ CHỐI**\n\nXác suất vỡ nợ: **{proba:.2%}**")
        else:
            st.success(f"**KHUYẾN NGHỊ: DUYỆT**\n\nXác suất vỡ nợ: **{proba:.2%}**")
    with c2:
        st.write("### 💼 Phân tích Lợi ích - Chi phí")
        exp_rev = loan_amnt * (int_rate / 100) * (1 - proba)
        exp_loss = loan_amnt * proba
        exp_profit = exp_rev - exp_loss
        m1, m2, m3 = st.columns(3)
        m1.metric("Doanh thu dự kiến", f"${exp_rev:,.2f}")
        m2.metric("Lỗ dự kiến", f"- ${exp_loss:,.2f}")
        if exp_profit > 0:
            m3.metric("LỢI NHUẬN ƯỚC TÍNH", f"${exp_profit:,.2f}", delta="Profit")
        else:
            m3.metric("LỖ ƯỚC TÍNH", f"${exp_profit:,.2f}", delta="- Loss", delta_color="inverse")

    st.markdown("---")
    st.write("### 🔍 Giải thích SHAP (Top 5 yếu tố)")
    try:
        explainer = get_shap_explainer(model, bg)
        sv = explainer.shap_values(X)[0]
        sdf = pd.DataFrame({"Feature": X.columns, "SHAP": sv,
                            "Val": X.iloc[0].values})
        sdf["Abs"] = sdf["SHAP"].abs()
        top = sdf.sort_values("Abs", ascending=False).head(5).sort_values("SHAP")
        colors = ["#ff0051" if v > 0 else "#008bfb" for v in top["SHAP"]]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(top["Feature"], top["SHAP"], color=colors)
        ax.axvline(0, color="black", lw=1, ls="--")
        ax.set_xlabel("Mức tác động đến rủi ro vỡ nợ")
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Không tạo được SHAP: {e}")

    st.markdown("---")
    st.write("### 🔄 What-if: Đề xuất hạn mức an toàn")
    if proba < 0.5:
        st.success("✅ Khoản vay yêu cầu đã ở ngưỡng an toàn.")
    else:
        safe_amt, safe_prob = find_safe_amount(model, raw, loan_amnt)
        if safe_amt:
            o1, o2 = st.columns(2)
            with o1:
                st.info(f"#### 💡 PA1: Hỗ trợ tối đa\n**Duyệt:** ${safe_amt:,.0f}\n\n"
                        f"Xác suất vỡ nợ: {safe_prob:.2%}")
            with o2:
                safer = int(safe_amt * 0.8)
                trial = dict(raw); trial["loan_amnt"] = safer
                st.success(f"#### 🛡️ PA2: An toàn cao\n**Duyệt:** ${safer:,.0f}\n\n"
                           f"Xác suất vỡ nợ: {predict_proba(model, trial):.2%}")
        else:
            st.error("❌ Rủi ro quá lớn; kể cả giảm về mức tối thiểu vẫn không thể tự duyệt.")


# GIAO DIỆN KHÁCH HÀNG (Self-service) — KHÔNG hiển thị SHAP / xác suất chi tiết
def customer_view(user, model, bg):
    st.title("🙋 Cổng Khách hàng — Tự kiểm tra khả năng vay")
    st.caption("Nhập thông tin để xem bạn có khả năng đủ điều kiện vay hay không, "
               "cùng gợi ý điều chỉnh phù hợp.")

    tab_check, tab_history = st.tabs(["🔎 Kiểm tra khoản vay", "🗂️ Hồ sơ của tôi"])

    with tab_check:
        st.write("### 📝 Thông tin khoản vay")
        raw, loan_amnt, int_rate = loan_input_form(key_prefix="cust_")

        with st.expander("Thông tin liên hệ (tùy chọn)"):
            phone = st.text_input("Số điện thoại", key="cust_phone")
            nid = st.text_input("CCCD/CMND", key="cust_nid")

        if st.button("Kiểm tra điều kiện vay", use_container_width=True):
            if model is None:
                st.error("Chưa có mô hình. Vui lòng chạy pipeline trước.")
                return

            proba = predict_proba(model, raw)
            eligible = proba < 0.5
            dti = pp.compute_dti(loan_amnt, raw["person_income"])

            st.markdown("---")
            if eligible:
                st.success(f"✅ **Tin tốt!** Với hồ sơ hiện tại, khoản vay "
                           f"**${loan_amnt:,.0f}** của bạn **có khả năng đủ điều kiện**.")
            else:
                st.warning("ℹ️ Khoản vay hiện tại **chưa đủ điều kiện tự động**. "
                           "Dưới đây là vài gợi ý giúp tăng khả năng được duyệt:")
                safe_amt, _ = find_safe_amount(model, raw, loan_amnt)
                if safe_amt:
                    st.info(f"💡 **Gợi ý 1:** Cân nhắc giảm số tiền vay xuống khoảng "
                            f"**${safe_amt:,.0f}** để tăng khả năng được duyệt.")
                else:
                    st.info("💡 **Gợi ý 1:** Hãy cân nhắc một khoản vay nhỏ hơn đáng kể.")
                if dti > 0.35:
                    st.info(f"💡 **Gợi ý 2:** Tỷ lệ Nợ/Thu nhập của bạn đang khá cao "
                            f"(**{dti:.2f}**). Giảm bớt khoản vay hoặc tăng thu nhập "
                            f"chứng minh sẽ giúp cải thiện hồ sơ.")
                st.caption("Đây là ước tính tham khảo. Quyết định cuối cùng do nhân "
                           "viên ngân hàng thẩm định.")

            # Lưu hồ sơ vào DB (owner = chính khách hàng -> phục vụ cô lập dữ liệu)
            app_id = db.save_application(
                owner=user["username"],
                applicant={"name": user["name"], "phone": phone,
                           "national_id": nid, "email": ""},
                raw_features=raw,
                result={"eligible": bool(eligible),
                        "self_service": True},   # không lưu xác suất chi tiết phía KH
            )
            st.toast(f"Đã gửi hồ sơ tới bộ phận thẩm định (mã {app_id}).")

    with tab_history:
        st.write("### 🗂️ Các hồ sơ bạn đã gửi")
        # CÔ LẬP DỮ LIỆU: customer chỉ nhận về hồ sơ của chính mình (lọc ở backend)
        apps = db.list_applications(user["username"], user["role"])
        if not apps:
            st.info("Bạn chưa gửi hồ sơ nào.")
        else:
            rows = [{
                "Mã": a["app_id"],
                "Số tiền vay": f"${a['features'].get('loan_amnt', 0):,.0f}",
                "Mục đích": a["features"].get("loan_intent", ""),
                "Trạng thái": a.get("decision", "PENDING"),
                "Ngày gửi": a.get("created_at", "")[:10],
            } for a in apps]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# GIAO DIỆN NHÂN VIÊN (Underwriter) — tách Historical vs New Applications
def staff_view(user, model, bg):
    st.title("👔 Cổng Nhân viên Thẩm định")

    tab_hist, tab_new, tab_manual = st.tabs([
        "📊 Historical Dashboard (Dữ liệu cũ)",
        "🆕 New Applications (Dữ liệu mới)",
        "🧮 Thẩm định thủ công",
    ])

    # TAB 1: Dashboard dữ liệu lịch sử (tách biệt hoàn toàn với hồ sơ mới)
    with tab_hist:
        st.write("### 📊 Tổng quan dữ liệu lịch sử")
        st.caption("🔒 Dashboard chỉ hiển thị số liệu tổng hợp/ẩn danh — "
                   "không hiển thị thông tin định danh khách hàng.")
        if os.path.exists(RAW_DATA_PATH):
            hist = pd.read_csv(RAW_DATA_PATH)
            c1, c2, c3 = st.columns(3)
            c1.metric("Tổng số hồ sơ", f"{len(hist):,}")
            if "loan_status" in hist:
                c2.metric("Tỷ lệ vỡ nợ", f"{hist['loan_status'].mean():.1%}")
            if "loan_amnt" in hist:
                c3.metric("Khoản vay TB", f"${hist['loan_amnt'].mean():,.0f}")

            if "loan_intent" in hist:
                st.write("**Phân bố theo mục đích vay**")
                st.bar_chart(hist["loan_intent"].value_counts())
            if "loan_status" in hist and "loan_intent" in hist:
                st.write("**Tỷ lệ vỡ nợ theo mục đích vay**")
                st.bar_chart(hist.groupby("loan_intent")["loan_status"].mean())
        else:
            st.warning("Không tìm thấy loan_data.csv.")

        if os.path.exists(RESULTS_PATH):
            st.write("**Hiệu suất các mô hình (từ Bước 5)**")
            st.dataframe(pd.read_csv(RESULTS_PATH), use_container_width=True, hide_index=True)

    # TAB 2: Hồ sơ mới khách hàng nộp (PII được che)
    with tab_new:
        st.write("### 🆕 Hồ sơ vay mới chờ thẩm định")
        apps = db.list_applications(user["username"], user["role"])  # staff -> xem tất cả
        if not apps:
            st.info("Chưa có hồ sơ mới nào được nộp.")
        else:
            # Bảng tóm tắt với PII đã CHE (Data Masking - Cải tiến 4)
            table = []
            for a in apps:
                ap = a.get("applicant", {})
                table.append({
                    "Mã": a["app_id"],
                    "Khách hàng": mask_name(ap.get("name", "")),
                    "SĐT": mask_phone(ap.get("phone", "")) if ap.get("phone") else "—",
                    "CCCD": mask_national_id(ap.get("national_id", "")) if ap.get("national_id") else "—",
                    "Số tiền vay": f"${a['features'].get('loan_amnt', 0):,.0f}",
                    "Trạng thái": a.get("decision", "PENDING"),
                })
            st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

            ids = [a["app_id"] for a in apps]
            chosen = st.selectbox("Chọn mã hồ sơ để thẩm định chi tiết", ids)
            app = next(a for a in apps if a["app_id"] == chosen)

            ap = app.get("applicant", {})
            st.caption(f"Khách hàng: **{mask_name(ap.get('name',''))}** · "
                       f"SĐT: {mask_phone(ap.get('phone','')) if ap.get('phone') else '—'}")

            if st.button("▶️ Chạy phân tích đầy đủ", use_container_width=True):
                if model is None:
                    st.error("Chưa có mô hình. Hãy chạy pipeline.")
                else:
                    raw = app["features"]
                    staff_full_analysis(model, bg, raw,
                                        raw.get("loan_amnt", 0),
                                        raw.get("loan_int_rate", 0))
                    d1, d2 = st.columns(2)
                    if d1.button("✅ Duyệt hồ sơ"):
                        db.set_decision(chosen, "APPROVED", user["username"])
                        st.success("Đã cập nhật: APPROVED"); st.rerun()
                    if d2.button("❌ Từ chối hồ sơ"):
                        db.set_decision(chosen, "REJECTED", user["username"])
                        st.error("Đã cập nhật: REJECTED"); st.rerun()

    # TAB 3: Thẩm định thủ công (nhập trực tiếp)
    with tab_manual:
        st.write("### 🧮 Nhập hồ sơ và thẩm định trực tiếp")
        raw, loan_amnt, int_rate = loan_input_form(key_prefix="staff_")
        if st.button("Phân tích", use_container_width=True, key="staff_analyze"):
            if model is None:
                st.error("Chưa có mô hình. Hãy chạy pipeline.")
            else:
                staff_full_analysis(model, bg, raw, loan_amnt, int_rate)


# Điều phối chính
def main():
    st.markdown("## 🏦 Personalized Credit Decision Support System")

    user = auth.require_login()
    if not user:
        st.stop()

    auth.render_sidebar_account(user)

    model = load_model()
    bg = load_background()
    if model is None:
        st.warning("⚠️ Chưa tìm thấy `stacking_ensemble_model.pkl`. "
                   "Hãy chạy `python main.py` để huấn luyện trước.")

    # PHÂN QUYỀN: điều hướng theo vai trò (Cải tiến 1)
    if user["role"] == db.ROLE_STAFF:
        staff_view(user, model, bg)
    else:
        customer_view(user, model, bg)


if __name__ == "__main__":
    main()
