import streamlit as st
import joblib
import numpy as np

# ==========================================
# Konfigurasi halaman
# ==========================================
st.set_page_config(
    page_title="Deteksi Kesehatan Urine",
    layout="centered"
)

# ==========================================
# Load model
# ==========================================
@st.cache_resource
def load_models():
    knn_model = joblib.load('model_knn.sav')
    nb_model = joblib.load('model_nb.sav')
    svm_model = joblib.load('model_svm.sav')
    scaler = joblib.load('scaler.sav')

    return knn_model, nb_model, svm_model, scaler

# ==========================================
# Main
# ==========================================
def main():
    st.title("🧪 Sistem Deteksi Kesehatan Urine")
    st.write("Masukkan parameter urine untuk analisis.")

    try:
        knn_model, nb_model, svm_model, scaler = load_models()

        with st.form("prediction_form"):

            col1, col2 = st.columns(2)

            with col1:
                ph = st.number_input(
                    "Nilai pH",
                    min_value=0.0,
                    max_value=14.0,
                    value=6.0,
                    step=0.1
                )

                r = st.number_input(
                    "Nilai Red (R)",
                    min_value=0,
                    max_value=255,
                    value=255
                )

            with col2:
                g = st.number_input(
                    "Nilai Green (G)",
                    min_value=0,
                    max_value=255,
                    value=255
                )

                b = st.number_input(
                    "Nilai Blue (B)",
                    min_value=0,
                    max_value=255,
                    value=0
                )

            submit = st.form_submit_button("Prediksi")

        if submit:

            input_data = np.array([[ph, r, g, b]])
            input_scaled = scaler.transform(input_data)

            hasil_knn = knn_model.predict(input_scaled)[0]
            hasil_nb = nb_model.predict(input_scaled)[0]
            hasil_svm = svm_model.predict(input_scaled)[0]

            st.divider()
            st.subheader("Hasil Analisis")

            if hasil_knn == 1:
                st.warning("Dehidrasi terdeteksi")
            else:
                st.success("Tidak terdeteksi Dehidrasi")

            if hasil_nb == 1:
                st.warning("Diabetes Mellitus terdeteksi")
            else:
                st.success("Tidak terdeteksi Diabetes Mellitus")

            if hasil_svm == 1:
                st.error("Gangguan Fungsi Ginjal terdeteksi")
            else:
                st.success("Tidak terdeteksi Gangguan Fungsi Ginjal")

            if hasil_knn == 0 and hasil_nb == 0 and hasil_svm == 0:
                st.balloons()
                st.success("Status urine normal")

    except FileNotFoundError:
        st.error("Pastikan semua file model ada di folder yang sama.")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")


if __name__ == "__main__":
    main()