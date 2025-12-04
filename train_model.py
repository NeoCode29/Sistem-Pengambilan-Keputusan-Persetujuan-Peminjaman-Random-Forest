import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load Data
print("Membaca dataset...")
df = pd.read_csv('data/loan_approval_dataset.csv')

# 2. Data Cleaning (PENTING: Menghapus spasi di nama kolom & isi data)
df.columns = df.columns.str.strip()
categorical_cols = ['education', 'self_employed', 'loan_status']
for col in categorical_cols:
    df[col] = df[col].str.strip()

# 3. Encoding (Mengubah teks menjadi angka agar bisa dihitung)
# Education: Not Graduate=0, Graduate=1
df['education'] = df['education'].map({'Not Graduate': 0, 'Graduate': 1})
# Self_Employed: No=0, Yes=1
df['self_employed'] = df['self_employed'].map({'No': 0, 'Yes': 1})
# Loan_Status: Rejected=0, Approved=1
df['loan_status'] = df['loan_status'].map({'Rejected': 0, 'Approved': 1})

# 4. Memisahkan Fitur (X) dan Target (y)
X = df.drop('loan_status', axis=1)  # Data input
y = df['loan_status']               # Target prediksi

# Hapus loan_id karena tidak berguna untuk prediksi
if 'loan_id' in X.columns:
    X = X.drop('loan_id', axis=1)

# 5. Bagi data untuk Latihan (Train) dan Ujian (Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Melatih Model (Random Forest)
print("Melatih model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluasi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi Model: {accuracy * 100:.2f}%")

# 8. Simpan Model
with open('loan_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model berhasil disimpan sebagai 'loan_model.pkl'")