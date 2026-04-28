import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# ==========================================
# 1. Load data
# ==========================================
df = pd.read_csv('dataset_urine.csv')

# ==========================================
# 2. Feature dan Target
# ==========================================
X = df[['pH', 'R', 'G', 'B']]

# Target binary per kondisi
y_dehidrasi = (df['Status'] == 'Dehidrasi').astype(int)
y_diabetes = (df['Status'] == 'Diabetes Mellitus').astype(int)
y_ginjal = (df['Status'] == 'Gangguan Fungsi Ginjal').astype(int)

# ==========================================
# 3. Scaling
# ==========================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Simpan scaler
joblib.dump(scaler, 'scaler.sav')

# ==========================================
# 4. Train KNN (Dehidrasi)
# ==========================================
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_scaled, y_dehidrasi)

# Simpan model KNN
joblib.dump(knn_model, 'model_knn.sav')

# ==========================================
# 5. Train Naive Bayes (Diabetes Mellitus)
# ==========================================
nb_model = GaussianNB()
nb_model.fit(X_scaled, y_diabetes)

# Simpan model Naive Bayes
joblib.dump(nb_model, 'model_nb.sav')

# ==========================================
# 6. Train SVM (Gangguan Fungsi Ginjal)
# ==========================================
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_scaled, y_ginjal)

# Simpan model SVM
joblib.dump(svm_model, 'model_svm.sav')

# ==========================================
# 7. Selesai
# ==========================================
print("Berhasil menyimpan:")
print("- model_knn.sav (Dehidrasi)")
print("- model_nb.sav (Diabetes Mellitus)")
print("- model_svm.sav (Gangguan Fungsi Ginjal)")
print("- scaler.sav")