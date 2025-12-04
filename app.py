import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI ---
st.set_page_config(page_title="Loan AI - Smart Decision", page_icon="🏦")

# --- LOAD MODEL ---
try:
    with open('loan_model_rf.pkl', 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        feature_names = data['features']
except FileNotFoundError:
    st.error("Model tidak ditemukan! Jalankan 'train_model.py' dulu.")
    st.stop()

# --- SIDEBAR INPUT ---
st.sidebar.header("📝 Data Pemohon")

def get_input():
    # Profil
    st.sidebar.subheader("Profil")
    dependents = st.sidebar.slider("Tanggungan", 0, 5, 2)
    edu = st.sidebar.selectbox("Pendidikan", ["Graduate", "Not Graduate"])
    emp = st.sidebar.selectbox("Wiraswasta", ["No", "Yes"])
    
    # Keuangan
    st.sidebar.subheader("Keuangan")
    income = st.sidebar.number_input("Gaji Tahunan", 1000000, 100000000, 10000000)
    cibil = st.sidebar.slider("CIBIL Score", 300, 900, 600)
    
    # Aset
    st.sidebar.subheader("Total Aset")
    asset_res = st.sidebar.number_input("Aset Rumah", 0, value=5000000)
    asset_com = st.sidebar.number_input("Aset Komersial", 0, value=0)
    asset_lux = st.sidebar.number_input("Aset Mewah", 0, value=0)
    asset_bank = st.sidebar.number_input("Aset Bank", 0, value=2000000)
    
    # Pinjaman
    st.sidebar.subheader("Pengajuan")
    loan_amt = st.sidebar.number_input("Jumlah Pinjaman", 1000000, 100000000, 5000000)
    term = st.sidebar.slider("Tenor (Bulan)", 2, 240, 12)
    
    # Encoding
    edu_v = 1 if edu == "Graduate" else 0
    emp_v = 1 if emp == "Yes" else 0
    
    data = {
        'no_of_dependents': dependents,
        'education': edu_v,
        'self_employed': emp_v,
        'income_annum': income,
        'loan_amount': loan_amt,
        'loan_term': term,
        'cibil_score': cibil,
        'residential_assets_value': asset_res,
        'commercial_assets_value': asset_com,
        'luxury_assets_value': asset_lux,
        'bank_asset_value': asset_bank
    }
    return pd.DataFrame(data, index=[0])

input_df = get_input()

# --- HALAMAN UTAMA ---
st.title("🏦 Analisa Kelayakan Kredit")
st.write("Sistem cerdas untuk memprediksi persetujuan pinjaman bank.")
st.info(f"Menganalisa data pemohon dengan Gaji Tahunan **Rp {input_df['income_annum'][0]:,}** dan Pinjaman **Rp {input_df['loan_amount'][0]:,}**")

if st.button("🔍 Analisa Sekarang"):
    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)
    
    # --- 1. TAMPILKAN KEPUTUSAN ---
    st.divider()
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if prediction[0] == 1:
            st.success("### ✅ KEPUTUSAN: DISETUJUI")
            st.write(f"Sistem yakin **{proba[0][1]*100:.1f}%** bahwa pinjaman ini aman.")
        else:
            st.error("### ❌ KEPUTUSAN: DITOLAK")
            st.write(f"Risiko terlalu tinggi. Probabilitas gagal bayar: **{proba[0][0]*100:.1f}%**")

    # --- 2. PENJELASAN (LOGIC BASED) ---
    with col2:
        st.metric("Skor CIBIL", input_df['cibil_score'][0])
        
    st.subheader("💡 Kenapa Keputusan ini diambil?")
    
    # Logika Penjelasan Sederhana untuk User
    cibil = input_df['cibil_score'][0]
    income = input_df['income_annum'][0]
    loan = input_df['loan_amount'][0]
    assets = (input_df['residential_assets_value'][0] + 
              input_df['commercial_assets_value'][0] + 
              input_df['luxury_assets_value'][0] + 
              input_df['bank_asset_value'][0])
    
    reasons = []
    
    # Analisa CIBIL
    if cibil < 550:
        reasons.append("❌ **Skor Kredit (CIBIL) Sangat Rendah.** Ini indikator utama penolakan. Bank butuh skor minimal 600-700.")
    elif cibil > 700:
        reasons.append("✅ **Skor Kredit Sangat Baik.** Riwayat kredit Anda memperbesar peluang persetujuan.")
        
    # Analisa Rasio Pinjaman vs Gaji
    if loan > income * 5:
        reasons.append("❌ **Pinjaman Terlalu Besar.** Jumlah pinjaman melebihi 5x gaji tahunan Anda. Ini dianggap berisiko.")
    elif loan < income * 2:
        reasons.append("✅ **Pinjaman Sehat.** Jumlah pinjaman masih dalam batas wajar dibanding pendapatan.")

    # Analisa Aset
    if assets < loan:
        reasons.append("⚠️ **Aset Kurang.** Total aset Anda lebih kecil dari pinjaman. Agunan mungkin kurang.")
        
    # Tampilkan Alasan
    if not reasons:
        st.write("Keputusan berdasarkan kombinasi pola data historis yang kompleks.")
    else:
        for r in reasons:
            st.write(r)

    # --- 3. GRAFIK FEATURE IMPORTANCE (DARI AI) ---
    st.divider()
    st.subheader("📊 Faktor Penentu Menurut Model Random Forrest")
    st.write("Grafik ini menunjukkan variabel apa yang paling dianggap penting oleh Model Komputer:")
    
    importances = model.feature_importances_
    feature_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    
    fig, ax = plt.subplots()
    sns.barplot(x=feature_imp.values, y=feature_imp.index, hue=feature_imp.index, legend=False, ax=ax, palette="viridis")
    ax.set_xlabel("Tingkat Kepentingan")
    st.pyplot(fig)