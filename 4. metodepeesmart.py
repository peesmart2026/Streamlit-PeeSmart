import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==================================================
# 1. Load Dataset
# ==================================================
df = pd.read_csv('dataset_urine.csv')

# ==================================================
# 2. Encoding Label
# ==================================================
le = LabelEncoder()
df['Status_Encoded'] = le.fit_transform(df['Status'])

# Feature
X = df[['pH', 'R', 'G', 'B']]

# ==================================================
# 3. Membuat Target Binary per Penyakit
# ==================================================

# Dehidrasi (KNN)
y_dehidrasi = (df['Status'] == 'Dehidrasi').astype(int)

# Diabetes Mellitus (Naive Bayes)
y_diabetes = (df['Status'] == 'Diabetes Mellitus').astype(int)

# Gangguan Fungsi Ginjal (SVM)
y_ginjal = (df['Status'] == 'Gangguan Fungsi Ginjal').astype(int)

# ==================================================
# 4. Split Data
# ==================================================

X_train, X_test, y_train_deh, y_test_deh = train_test_split(
    X, y_dehidrasi, test_size=0.2, random_state=42
)

_, _, y_train_dm, y_test_dm = train_test_split(
    X, y_diabetes, test_size=0.2, random_state=42
)

_, _, y_train_ginjal, y_test_ginjal = train_test_split(
    X, y_ginjal, test_size=0.2, random_state=42
)

# ==================================================
# 5. Normalisasi Data
# ==================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================================================
# 6. Model KNN (Dehidrasi)
# ==================================================
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train_deh)

y_pred_knn = knn_model.predict(X_test_scaled)

print("\n=== HASIL KNN (DEHIDRASI) ===")
print(f"Accuracy: {accuracy_score(y_test_deh, y_pred_knn)*100:.2f}%")
print(classification_report(y_test_deh, y_pred_knn))

# Confusion Matrix KNN
cm_knn = confusion_matrix(y_test_deh, y_pred_knn)
plt.figure(figsize=(6,4))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - KNN (Dehidrasi)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ==================================================
# 7. Model Naive Bayes (Diabetes Mellitus)
# ==================================================
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train_dm)

y_pred_nb = nb_model.predict(X_test_scaled)

print("\n=== HASIL NAIVE BAYES (DIABETES MELLITUS) ===")
print(f"Accuracy: {accuracy_score(y_test_dm, y_pred_nb)*100:.2f}%")
print(classification_report(y_test_dm, y_pred_nb))

# Confusion Matrix NB
cm_nb = confusion_matrix(y_test_dm, y_pred_nb)
plt.figure(figsize=(6,4))
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix - Naive Bayes (Diabetes Mellitus)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ==================================================
# 8. Model SVM (Gangguan Fungsi Ginjal)
# ==================================================
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_scaled, y_train_ginjal)

y_pred_svm = svm_model.predict(X_test_scaled)

print("\n=== HASIL SVM (GANGGUAN FUNGSI GINJAL) ===")
print(f"Accuracy: {accuracy_score(y_test_ginjal, y_pred_svm)*100:.2f}%")
print(classification_report(y_test_ginjal, y_pred_svm))

# Confusion Matrix SVM
cm_svm = confusion_matrix(y_test_ginjal, y_pred_svm)
plt.figure(figsize=(6,4))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Reds')
plt.title('Confusion Matrix - SVM (Gangguan Fungsi Ginjal)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()