import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load Data
print("Membaca dataset...")
df = pd.read_csv('./data/loan_approval_dataset.csv')

# 2. Cleaning
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# 3. Encoding
df['education'] = df['education'].map({'Not Graduate': 0, 'Graduate': 1})
df['self_employed'] = df['self_employed'].map({'No': 0, 'Yes': 1})
df['loan_status'] = df['loan_status'].map({'Rejected': 0, 'Approved': 1})

# 4. Split Data
X = df.drop(['loan_id', 'loan_status'], axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Latih Model Random Forest
print("Melatih model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluasi
acc = accuracy_score(y_test, model.predict(X_test))
print(f"Akurasi: {acc*100:.2f}%")

# 7. Simpan Model & Nama Kolom (PENTING untuk visualisasi)
feature_names = X.columns.tolist()
with open('loan_model_rf.pkl', 'wb') as f:
    pickle.dump({'model': model, 'features': feature_names}, f)

print("Model berhasil disimpan ke 'loan_model_rf.pkl'")