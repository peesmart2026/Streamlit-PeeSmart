import joblib
import numpy as np

# ==========================================
# Load semua model
# ==========================================
knn_model = joblib.load('model_knn.sav')
nb_model = joblib.load('model_nb.sav')
svm_model = joblib.load('model_svm.sav')
scaler = joblib.load('scaler.sav')

# ==========================================
# Contoh data baru
# Format: [pH, R, G, B]
# ==========================================
data_baru = np.array([[6.0, 255.0, 255.0, 0.0]])

# Scaling data
data_scaled = scaler.transform(data_baru)

# ==========================================
# Prediksi masing-masing model
# ==========================================
hasil_knn = knn_model.predict(data_scaled)
hasil_nb = nb_model.predict(data_scaled)
hasil_svm = svm_model.predict(data_scaled)

# ==========================================
# Tampilkan hasil
# ==========================================
print("=== HASIL PREDIKSI ===")

if hasil_knn[0] == 1:
    print("Dehidrasi: Terdeteksi")
else:
    print("Dehidrasi: Tidak Terdeteksi")

if hasil_nb[0] == 1:
    print("Diabetes Mellitus: Terdeteksi")
else:
    print("Diabetes Mellitus: Tidak Terdeteksi")

if hasil_svm[0] == 1:
    print("Gangguan Fungsi Ginjal: Terdeteksi")
else:
    print("Gangguan Fungsi Ginjal: Tidak Terdeteksi")

# Status normal jika semua negatif
if hasil_knn[0] == 0 and hasil_nb[0] == 0 and hasil_svm[0] == 0:
    print("Status: Normal")