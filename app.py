import streamlit as st
import pandas as pd
import pickle

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Loan Approval System",
    page_icon="💰",
    layout="centered"
)

# --- Judul & Deskripsi ---
st.title("💰 Prediksi Kelayakan Pinjaman")
st.write("Aplikasi ini menggunakan Machine Learning untuk memprediksi apakah pengajuan pinjaman akan **Disetujui** atau **Ditolak**.")

# --- Load Model ---
try:
    with open('loan_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model belum ditemukan! Jalankan 'train_model.py' terlebih dahulu.")
    st.stop()

# --- Input User (Sidebar) ---
st.sidebar.header("📝 Masukkan Data Pemohon")

def user_input_features():
    # 1. Data Profil
    st.sidebar.subheader("Profil")
    no_of_dependents = st.sidebar.slider("Jumlah Tanggungan", 0, 5, 1)
    education = st.sidebar.selectbox("Pendidikan", ["Graduate", "Not Graduate"])
    self_employed = st.sidebar.selectbox("Wiraswasta?", ["No", "Yes"])
    
    # 2. Data Keuangan
    st.sidebar.subheader("Keuangan")
    income_annum = st.sidebar.number_input("Pendapatan Tahunan (IDR)", min_value=0, value=50000000, step=1000000)
    cibil_score = st.sidebar.slider("CIBIL Score (Kredit Skor)", 300, 900, 600)
    
    # 3. Data Aset
    st.sidebar.subheader("Aset")
    residential_assets = st.sidebar.number_input("Nilai Aset Rumah", min_value=0, value=0)
    commercial_assets = st.sidebar.number_input("Nilai Aset Komersial", min_value=0, value=0)
    luxury_assets = st.sidebar.number_input("Nilai Aset Mewah", min_value=0, value=0)
    bank_asset = st.sidebar.number_input("Nilai Aset di Bank", min_value=0, value=0)
    
    # 4. Data Pinjaman
    st.sidebar.subheader("Pengajuan Pinjaman")
    loan_amount = st.sidebar.number_input("Jumlah Pinjaman Diajukan", min_value=0, value=10000000)
    loan_term = st.sidebar.slider("Jangka Waktu (Bulan)", 0, 240, 12)

    # Konversi input ke format angka (sesuai training)
    education_val = 1 if education == "Graduate" else 0
    self_employed_val = 1 if self_employed == "Yes" else 0
    
    data = {
        'no_of_dependents': no_of_dependents,
        'education': education_val,
        'self_employed': self_employed_val,
        'income_annum': income_annum,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'cibil_score': cibil_score,
        'residential_assets_value': residential_assets,
        'commercial_assets_value': commercial_assets,
        'luxury_assets_value': luxury_assets,
        'bank_asset_value': bank_asset
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- Tampilkan Input User ---
st.subheader("📋 Data yang Anda Masukkan")
st.write(input_df)

# --- Tombol Prediksi ---
if st.button("🔍 Analisa Kelayakan"):
    # Prediksi
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    
    st.subheader("📊 Hasil Prediksi")
    
    if prediction[0] == 1:
        st.success("✅ **SELAMAT! Pinjaman Kemungkinan Besar DISETUJUI**")
        st.balloons()
    else:
        st.error("❌ **MAAF, Pinjaman Kemungkinan Besar DITOLAK**")
    
    st.write(f"Probabilitas Disetujui: **{probability[0][1] * 100:.2f}%**")
    st.write(f"Probabilitas Ditolak: **{probability[0][0] * 100:.2f}%**")