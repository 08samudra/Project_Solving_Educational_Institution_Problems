import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# === Load model & data ===
df = pd.read_csv("datasets/data_preprocessed.csv")
model = joblib.load("models/best_random_forest.pkl")
feature_names = joblib.load("models/rf_feature_names.pkl")

st.set_page_config(page_title="🎓 Dashboard Prediksi Dropout", layout="wide")

# ==== Navigasi Menu ====
menu = st.sidebar.radio("📂 Navigasi Menu", [
    "1. Distribusi Data",
    "2. Tren Risiko Dropout per Fitur",
    "3. Visualisasi Faktor Dropout",
    "4. Daftar Mahasiswa Aktif Berisiko",
    "5. Prediksi Manual Mahasiswa Aktif"
])

# ========== MENU 1 ==========
if menu == "1. Distribusi Data":
    st.title("🎓 Jaya Jaya Institute – Analisis & Visualisasi Prediktif")
    st.title("📊 Distribusi Data Mahasiswa")
    st.markdown("""

Memberikan gambaran umum mengenai komposisi dan karakteristik awal data mahasiswa.

**Isi Menu:**
- **Distribusi Status Mahasiswa**: Menampilkan jumlah mahasiswa berdasarkan tiga status utama: *Dropout, Enrolled, Graduate*.  
- **Usia Saat Mendaftar**: Histogram usia mahasiswa saat masuk perguruan tinggi.  
- **Statistik Fitur Numerik**: Tabel `describe()` untuk melihat mean, min, max, dan standar deviasi fitur numerik.

📌 Menu ini menjadi fondasi eksplorasi dan pemahaman dataset sebelum modeling dilakukan.
""")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Status Mahasiswa")
        st.bar_chart(df["Status"].value_counts())

    with col2:
        st.subheader("Usia Saat Mendaftar")
        fig, ax = plt.subplots()
        sns.histplot(df["Age_at_enrollment"], kde=True, bins=20, color="slateblue", ax=ax)
        st.pyplot(fig)

    st.subheader("📋 Statistik Ringkas Fitur Numerik")
    st.dataframe(df.describe().T)

# ========== MENU 2 ==========
elif menu == "2. Tren Risiko Dropout per Fitur":
    st.title("📉 Tren Risiko Dropout per Fitur")
    st.markdown("""

 Menelusuri hubungan antara nilai fitur tertentu dan kemungkinan dropout berdasarkan observasi di dataset.

**Isi Menu:**
- **Interaktif**: pengguna memilih fitur penting dari dropdown.
- **Visualisasi**:
    - Jika kategorikal → *barplot* rata-rata dropout per kategori.
    - Jika numerik → *lineplot* tren dropout terhadap nilai fitur.

**Contoh Fitur:**
- Admission_grade → apakah nilai masuk rendah cenderung dropout?
- Tuition_fees_up_to_date → apakah pembayaran biaya kuliah mempengaruhi dropout?
- Curricular_units_1st_sem_grade → apakah nilai awal menjadi indikator risiko?

📌 Menu ini membantu stakeholder membaca pola risiko berdasarkan fitur akademik maupun administratif.
""")

    selected_feature = st.selectbox("📌 Pilih Fitur:", [
        "Curricular_units_2nd_sem_approved",
        "Curricular_units_1st_sem_grade",
        "Tuition_fees_up_to_date",
        "Admission_grade",
        "Age_at_enrollment"
    ])

    fig, ax = plt.subplots()
    if df[selected_feature].nunique() <= 5:
        sns.barplot(x=selected_feature, y="Dropout_Label", data=df, estimator=np.mean, ax=ax, palette="coolwarm")
        ax.set_ylabel("Probabilitas Dropout")
    else:
        sns.lineplot(x=selected_feature, y="Dropout_Label", data=df, estimator=np.mean, ax=ax, marker="o")
        ax.set_ylabel("Rata-rata Dropout")

    plt.title(f"Risiko Dropout berdasarkan {selected_feature}")
    st.pyplot(fig)

# ========== MENU 3 ==========
elif menu == "3. Visualisasi Faktor Dropout":
    st.title("📈 Visualisasi Faktor Dropout")
    st.markdown("""

Menyajikan visualisasi mendalam dari fitur-fitur yang berpengaruh besar pada prediksi dropout.

**Isi Menu:**
- **Boxplot MK Semester 1** → Dropout cenderung memiliki lebih sedikit MK yang disetujui.
- **Countplot Status Pembayaran** → Mahasiswa dropout lebih banyak menunggak.
- **Distribusi Admission Grade** → Visual perbedaan akademik awal antara yang dropout dan tidak.

📌 Visual-visual ini dapat menjadi dasar rancangan intervensi atau kebijakan kampus.
""")

    st.subheader("MK Disetujui Semester 1")
    fig1, ax1 = plt.subplots()
    sns.boxplot(x="Dropout_Label", y="Curricular_units_1st_sem_approved", data=df, ax=ax1, palette="Set2")
    ax1.set_xticklabels(["Tidak Dropout", "Dropout"])
    st.pyplot(fig1)

    st.subheader("Status Pembayaran Biaya Kuliah")
    fig2, ax2 = plt.subplots()
    sns.countplot(x="Dropout_Label", hue="Tuition_fees_up_to_date", data=df, palette="coolwarm", ax=ax2)
    ax2.set_xticklabels(["Tidak Dropout", "Dropout"])
    ax2.legend(title="Biaya Terbayar")
    st.pyplot(fig2)

    st.subheader("Distribusi Nilai Masuk")
    fig3, ax3 = plt.subplots()
    sns.kdeplot(df[df["Dropout_Label"] == 0]["Admission_grade"], label="Tidak Dropout", fill=True)
    sns.kdeplot(df[df["Dropout_Label"] == 1]["Admission_grade"], label="Dropout", fill=True, color="red")
    ax3.legend()
    st.pyplot(fig3)

# ========== MENU 4 ==========
elif menu == "4. Daftar Mahasiswa Aktif Berisiko":
    st.title("📋 Daftar Mahasiswa Aktif Berisiko Dropout")
    st.markdown("""
                
Menampilkan mahasiswa **Enrolled** dengan risiko dropout tertinggi berdasarkan hasil prediksi model.

**Isi Menu:**
- Prediksi batch pada seluruh mahasiswa aktif.
- Probabilitas dropout dihitung dan disortir.
- Ditampilkan top-N mahasiswa dengan risiko tertinggi.

**Informasi yang Ditampilkan:**
- Usia saat mendaftar
- Nilai admission
- MK disetujui semester 1
- Status pembayaran
- Probabilitas dropout (dalam persentase)

📌 Menu ini adalah sistem peringatan dini untuk membantu kampus intervensi sebelum terlambat.
""")

    df_enrolled = df[df["Status"] == "Enrolled"].copy()
    X_enrolled = df_enrolled[feature_names]
    df_enrolled["Prob_Dropout"] = model.predict_proba(X_enrolled)[:, 1]
    df_sorted = df_enrolled.sort_values("Prob_Dropout", ascending=False)

    top_k = st.slider("🔢 Tampilkan top-k mahasiswa", 5, 50, 10)
    st.dataframe(
        df_sorted[["Age_at_enrollment", "Admission_grade", "Curricular_units_1st_sem_approved", "Tuition_fees_up_to_date", "Prob_Dropout"]]
        .head(top_k)
        .style.format({"Prob_Dropout": "{:.2%}"})
    )

# ========== MENU 5 ==========
elif menu == "5. Prediksi Manual Mahasiswa Aktif":
    st.title("🧠 Prediksi Manual untuk Mahasiswa Aktif")
    st.markdown("""

Simulasikan prediksi risiko dropout berdasarkan input karakteristik mahasiswa aktif.

**Cara Menggunakan:**
1. Sistem menampilkan profil mahasiswa Enrolled sebagai baseline.
2. Pengguna bisa mengubah:
   - Usia, nilai masuk, nilai semester 1 & 2
   - Jumlah MK yang disetujui
   - Status pembayaran & tunggakan
3. Klik “🔍 Prediksi Dropout” untuk melihat hasil.

**Hasil:**
- Probabilitas dropout (%)
- Label klasifikasi: ❗ Dropout atau ✅ Tidak Dropout
- Tabel input yang digunakan model untuk prediksi

📌 Cocok digunakan oleh tim akademik untuk mengevaluasi risiko individual secara cepat.
""")

    enrolled_sample = df[df["Status"] == "Enrolled"].sample(1, random_state=42)[feature_names].iloc[0].copy()
    input_df = pd.DataFrame([enrolled_sample])

    age = st.slider("Usia Saat Mendaftar", 17, 70, int(input_df["Age_at_enrollment"]))
    admission_grade = st.slider("Nilai Masuk (Admission Grade)", 95.0, 190.0, float(input_df["Admission_grade"]))
    approved_1st = st.slider("MK Disetujui Semester 1", 0, 30, int(input_df["Curricular_units_1st_sem_approved"]))
    approved_2nd = st.slider("MK Disetujui Semester 2", 0, 30, int(input_df["Curricular_units_2nd_sem_approved"]))
    grade_1st = st.slider("Nilai Semester 1", 0.0, 20.0, float(input_df["Curricular_units_1st_sem_grade"]))
    grade_2nd = st.slider("Nilai Semester 2", 0.0, 20.0, float(input_df["Curricular_units_2nd_sem_grade"]))
    debtor = st.selectbox("Memiliki Tunggakan?", ["Ya", "Tidak"], index=0 if input_df["Debtor"].item() == 1 else 1)
    tuition = st.selectbox("Biaya Kuliah Terbayar?", ["Ya", "Tidak"], index=0 if input_df["Tuition_fees_up_to_date"].item() == 1 else 0)

    # Update input data
    input_df["Age_at_enrollment"] = age
    input_df["Admission_grade"] = admission_grade
    input_df["Curricular_units_1st_sem_approved"] = approved_1st
    input_df["Curricular_units_2nd_sem_approved"] = approved_2nd
    input_df["Curricular_units_1st_sem_grade"] = grade_1st
    input_df["Curricular_units_2nd_sem_grade"] = grade_2nd
    input_df["Debtor"] = 1 if debtor == "Ya" else 0
    input_df["Tuition_fees_up_to_date"] = 1 if tuition == "Ya" else 0

    # Prediksi
    if st.button("🔍 Prediksi Dropout"):
        prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]

        st.write(f"🧮 Probabilitas Dropout: **{prob:.2%}**")
        if pred == 1:
            st.error("⚠️ Mahasiswa ini diprediksi **berisiko dropout**.")
        else:
            st.success("✅ Mahasiswa ini **tidak diprediksi dropout** saat ini.")

        st.markdown("### 🔎 Data Mahasiswa yang Diprediksi")
        st.dataframe(input_df.T)
