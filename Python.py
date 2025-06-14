# %% [markdown]
# # Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

# %% [markdown]
# - Nama: Yoga Samudra
# - Email: 08samudra@gmail.com
# - Id Dicoding: 08samudra

# %% [markdown]
# ## 1. Persiapan

# %% [markdown]
# ### 1.1 Menyiapkan library yang dibutuhkan

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# %% [markdown]
# ### 1.2 Menyiapkan data yang akan diguankan

# %%
# Load dataset
file_path = "datasets/data.csv"
df = pd.read_csv(file_path, delimiter=';')

# Menampilkan informasi dasar tentang dataset
print("Dimensi Data:", df.shape)
print("\nInfo Dataset:")
print(df.info())

# Melihat beberapa data pertama
print("\nPreview Data:")
print(df.head())

# %% [markdown]
# #### **Struktur Dataset**
# Dataset memiliki beberapa variabel yang dapat dikategorikan sebagai berikut:
# 
# ##### **1. Informasi Demografi**
# - `Marital_status` : Status pernikahan mahasiswa *(1 - single, 2 - married, dst.)*
# - `Nationality` : Kewarganegaraan mahasiswa *(1 - Portuguese, 2 - German, dst.)*
# - `Gender` : Jenis kelamin mahasiswa *(1 - male, 0 - female)*
# - `Age_at_enrollment` : Usia mahasiswa saat mendaftar *(Numerik)*
# - `Displaced` : Apakah mahasiswa merupakan orang yang terdampak *(1 - yes, 0 - no)*
# 
# ##### **2. Informasi Akademik**
# - `Application_mode` : Metode pendaftaran yang digunakan *(1 - 1st phase - general contingent, dst.)*
# - `Application_order` : Urutan aplikasi pendaftaran mahasiswa *(0 - pertama pilihan, 9 - pilihan terakhir)*
# - `Previous_qualification` : Pendidikan sebelumnya *(1 - Secondary education, dst.)*
# - `Previous_qualification_grade` : Nilai pendidikan sebelumnya *(Skala 0 - 200)*
# - `Admission_grade` : Nilai penerimaan mahasiswa *(Skala 0 - 200)*
# - `Course` : Jurusan yang diambil mahasiswa *(33 - Biofuel Production Technologies, dst.)*
# - `Daytime_evening_attendance` : Kelas pagi atau malam *(1 - daytime, 0 - evening)*
# - `Educational_special_needs` : Apakah mahasiswa memiliki kebutuhan khusus *(1 - yes, 0 - no)*
# 
# ##### **3. Informasi Ekonomi dan Sosial**
# - `Mothers_qualification` & `Fathers_qualification` : Pendidikan orang tua *(1 - Secondary Education, dst.)*
# - `Mothers_occupation` & `Fathers_occupation` : Pekerjaan orang tua *(0 - Student, dst.)*
# - `Debtor` : Apakah mahasiswa memiliki tunggakan *(1 - yes, 0 - no)*
# - `Tuition_fees_up_to_date` : Apakah pembayaran kuliah mahasiswa lancar *(1 - yes, 0 - no)*
# - `Scholarship_holder` : Apakah mahasiswa mendapatkan beasiswa *(1 - yes, 0 - no)*
# 
# ##### **4. Informasi Akademik Semesteran**
# - `Curricular_units_1st_sem_credited`, `Curricular_units_1st_sem_enrolled`, `Curricular_units_1st_sem_evaluations`, `Curricular_units_1st_sem_approved`, `Curricular_units_1st_sem_grade`, `Curricular_units_1st_sem_without_evaluations`
# - `Curricular_units_2nd_sem_credited`, `Curricular_units_2nd_sem_enrolled`, `Curricular_units_2nd_sem_evaluations`, `Curricular_units_2nd_sem_approved`, `Curricular_units_2nd_sem_grade`, `Curricular_units_2nd_sem_without_evaluations`
# 
# ##### **5. Indikator Ekonomi**
# - `Unemployment_rate` : Tingkat pengangguran di wilayah mahasiswa *(Persentase)*
# - `Inflation_rate` : Tingkat inflasi ekonomi saat mahasiswa mendaftar *(Persentase)*
# - `GDP` : Produk Domestik Bruto sebagai indikator ekonomi *(Numerik)*
# 
# ##### **6. Target Variabel**
# - `Status` : Status akhir mahasiswa *(Dropout, Enrolled, atau Graduate)*
# 
# ---

# %% [markdown]
# ### 1.3 Penanganan *Missing Values* dan Data Duplikat

# %% [markdown]
# Cek *Missing Values*

# %%
# Mengecek missing values
print(df.isnull().sum())

# %% [markdown]
# Hasil cek dari *missing values* adalah 0

# %% [markdown]
# Cek Data Duplikat

# %%
# Mengecek jumlah data duplikat
print("Jumlah Data Duplikat:", df.duplicated().sum())

# %% [markdown]
# Hasil cek dari data duplikat adalah 0

# %% [markdown]
# ## 2. Data Understanding

# %% [markdown]
# ### 2.1 Statistik Deskriptif
# 

# %% [markdown]
# Memeriksa distribusi data numerik dan kategorikal

# %%
# Statistik deskriptif untuk data numerik
stats_numerik = df.describe().T

text_stats_numerik = "\n📊 Statistik Deskriptif untuk Data Numerik:\n"
for col in stats_numerik.index:
    text_stats_numerik += f"- {col}: Min = {stats_numerik.loc[col, 'min']}, Max = {stats_numerik.loc[col, 'max']}, Mean = {stats_numerik.loc[col, 'mean']:.2f}, Std = {stats_numerik.loc[col, 'std']:.2f}\n"

# Statistik deskriptif untuk data kategorikal
stats_kategorikal = df.select_dtypes(include=['object']).describe().T

text_stats_kategorikal = "\n📊 Statistik Deskriptif untuk Data Kategorikal:\n"
for col in stats_kategorikal.index:
    text_stats_kategorikal += f"- {col}: Jumlah Unik = {stats_kategorikal.loc[col, 'unique']}, Nilai Paling Sering = {stats_kategorikal.loc[col, 'top']} ({stats_kategorikal.loc[col, 'freq']} kali muncul)\n"

# Gabungkan teks
text_stats = text_stats_numerik + text_stats_kategorikal
print(text_stats)

# %% [markdown]
# ### 2.2 Distribusi *Status* (Dropout, Enrolled, Graduate)

# %% [markdown]
# Melihat pola dropout.

# %%
# Distribusi Status Mahasiswa
sns.countplot(x='Status', data=df)
plt.title('Distribusi Status Mahasiswa')
plt.show()

# %%
# Hitung jumlah mahasiswa per kategori Status
status_counts = df["Status"].value_counts()

text_status = "\n📊 Distribusi Status Mahasiswa:\n"
for status, count in status_counts.items():
    text_status += f"- {status}: {count} mahasiswa\n"

print(text_status)

# %% [markdown]
# ### 2.3 Distribusi Fitur-Fitur Utama

# %% [markdown]
# Distribusi Nilai Admission Grade Berdasarkan Status

# %%
sns.boxplot(x='Status', y='Admission_grade', data=df)
plt.title('Distribusi Admission Grade Berdasarkan Status Mahasiswa')
plt.show()

# %% [markdown]
# Distribusi Usia Saat Pendaftaran

# %%
sns.histplot(df['Age_at_enrollment'], bins=20, kde=True)
plt.title('Distribusi Usia Saat Pendaftaran Mahasiswa')
plt.xlabel('Usia')
plt.ylabel('Jumlah Mahasiswa')
plt.show()

# %% [markdown]
# Analisis Pengaruh Beasiswa Terhadap Dropout

# %%
sns.countplot(x='Status', hue='Scholarship_holder', data=df)
plt.title('Pengaruh Beasiswa terhadap Status Mahasiswa')
plt.xlabel('Status')
plt.ylabel('Jumlah Mahasiswa')
plt.legend(['Tanpa Beasiswa', 'Penerima Beasiswa'])
plt.show()

# %%
# Statistik Admission Grade berdasarkan Status
admission_stats = df.groupby("Status")["Admission_grade"].describe()

text_admission = "\n📊 Distribusi Admission Grade Berdasarkan Status:\n"
for status in admission_stats.index:
    text_admission += f"- {status}: Min = {admission_stats.loc[status, 'min']}, Max = {admission_stats.loc[status, 'max']}, Mean = {admission_stats.loc[status, 'mean']:.2f}, Std = {admission_stats.loc[status, 'std']:.2f}\n"

# Statistik Usia saat Pendaftaran
age_stats = df["Age_at_enrollment"].describe()

text_age = "\n📊 Distribusi Usia Saat Pendaftaran:\n"
text_age += f"- Min = {age_stats['min']}, Max = {age_stats['max']}, Mean = {age_stats['mean']:.2f}, Std = {age_stats['std']:.2f}\n"

# Statistik Pengaruh Beasiswa terhadap Status
scholarship_counts = df.groupby("Status")["Scholarship_holder"].value_counts()

text_scholarship = "\n📊 Pengaruh Beasiswa terhadap Status:\n"
for (status, scholarship), count in scholarship_counts.items():
    beasiswa_text = "Penerima Beasiswa" if scholarship == 1 else "Tanpa Beasiswa"
    text_scholarship += f"- {status} ({beasiswa_text}): {count} mahasiswa\n"

# Gabungkan teks
text_distribution = text_admission + text_age + text_scholarship
print(text_distribution)

# %% [markdown]
# ## 3. Data Preparation / Preprocessing

# %% [markdown]
# ### 3.1 Mengatasi variabel kategorikal
# 

# %% [markdown]
# Variabel kategorikal dalam dataset harus dikonversi ke format numerik agar dapat digunakan dalam model machine learning. Ada beberapa metode yang bisa digunakan:
# 
# 1. **Label Encoding** → Cocok untuk variabel yang memiliki urutan logis.
# 2. **One-Hot Encoding** → Cocok untuk variabel kategorikal tanpa urutan tertentu.
# 3. **Mapping Manual** → Cocok untuk variabel dengan sedikit kategori.

# %% [markdown]
# 1. Label Encoding (Variabel dengan Urutan):
# 
#     Metode ini mengonversi kategori menjadi angka, cocok untuk variabel yang memiliki urutan tertentu seperti *Marital Status* atau *Previous Qualification*.

# %%
# Daftar kolom yang akan di-label encoding
label_cols = ["Marital_status", "Daytime_evening_attendance", "Debtor", "Tuition_fees_up_to_date", "Gender", "Scholarship_holder"]

# Melakukan Label Encoding
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Menyimpan encoder untuk nanti digunakan kembali jika perlu


# %% [markdown]
# 2. One-Hot Encoding (Variabel Tanpa Urutan):
# 
#     Metode ini akan membuat kolom tambahan untuk setiap kategori yang ada, lalu kita menghapus satu kategori (drop_first=True) untuk menghindari multikolinearitas.

# %%
df = pd.get_dummies(df, columns=["Course", "Nacionality", "Mothers_qualification", "Fathers_qualification"], drop_first=True)


# %% [markdown]
# 3. Mapping Manual (Variabel dengan Sedikit Kategori):
# 
#     Jika hanya ada beberapa kategori seperti variabel Status, kita bisa menggunakan metode manual.

# %%
status_mapping = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
df["Status_Numeric"] = df["Status"].map(status_mapping)

# %% [markdown]
# ### 3.2 Normalisasi Variabel Numerik
# 
# Normalisasi bertujuan untuk menyamakan skala semua fitur numerik, sehingga tidak ada fitur yang mendominasi hanya karena memiliki nilai yang lebih besar. Ini sangat penting untuk algoritma seperti K-NN, SVM, dan Gradient Descent-based models (seperti logistic regression, neural network, dll).

# %% [markdown]
# Menentukan Fitur yang Akan Dinormalisasi:
# 
# Kita akan mengekstrak semua kolom numerik, lalu menghindari normalisasi pada target (*Status_Numeric*) karena itu adalah label.

# %%
# Memilih semua fitur numerik KECUALI kolom target
numerical_cols = df.select_dtypes(include=['number']).drop(columns=['Status_Numeric']).columns

# Inisialisasi MinMaxScaler
scaler = MinMaxScaler()

# Lakukan normalisasi
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# %% [markdown]
# ### 3.3 Pembagian Data *Train* dan *Test*
# 
# Sebelum memulai proses pelatihan model, dataset harus dibagi menjadi dua bagian:
# - Data *Training* → Untuk melatih model agar belajar dari pola data.
# - Data *Testing* → Untuk mengevaluasi performa model pada data yang belum pernah dilihat sebelumnya.

# %% [markdown]
# - Pisahkan Fitur dan Target
# 
#     Pastikan kita tidak menyertakan kolom `Status` asli yang masih berupa string, dan gunakan `Status_Numeric` sebagai label prediksi.

# %%
# X = fitur (selain Status dan Status_Numeric jika tidak relevan)
X = df.drop(columns=["Status", "Status_Numeric"])

# y = target (Status Numerik untuk prediksi dropout)
y = df["Status_Numeric"]

# %% [markdown]
# - Bagi Data dengan Stratifikasi
# 
#     Gunakan stratifikasi untuk menjaga proporsi kelas tetap seimbang di data training dan testing.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # menjaga proporsi label yang seimbang
)


# %% [markdown]
# Hasil *Train* dan *Test*

# %%
print(f"Jumlah data train: {X_train.shape[0]}")
print(f"Jumlah data test : {X_test.shape[0]}")
print("\nDistribusi kelas pada data train:")
print(y_train.value_counts(normalize=True))


# %% [markdown]
# ### 3.4 Reframing: Prediksi Risiko Dropout (Binary Classification)

# %% [markdown]
# Pada tahap ini, kita mengubah skenario prediksi dari tiga kelas (Dropout, Enrolled, Graduate) menjadi dua kelas saja:
# - **1 = Mahasiswa Dropout (berisiko keluar)**
# - **0 = Mahasiswa Tidak Dropout (baik masih Enrolled maupun sudah Graduate)**
# 
# Tujuannya adalah membangun model untuk mengenali **mahasiswa yang berisiko dropout**, sehingga sistem dapat memberikan _early warning_ kepada pihak akademik. Dengan pendekatan ini, kita lebih fokus pada penanganan dan intervensi untuk mahasiswa yang masih aktif.

# %%
# 3.4 Reframing: Binary Classification untuk Prediksi Dropout
df["Dropout_Label"] = df["Status"].apply(lambda x: 1 if x == "Dropout" else 0)

# Gunakan semua fitur selain label asli dan kolom string status
X = df.drop(columns=["Status", "Status_Numeric", "Dropout_Label"])
y = df["Dropout_Label"]

# Bagi data dengan stratifikasi
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# %% [markdown]
# Simpan Data

# %%
# Simpan dataset yang sudah diproses penuh (termasuk encoding dan normalisasi)
df.to_csv("data_preprocessed.csv", index=False)

# %%
# Setelah selesai training & preprocessing
feature_names = X_train.columns.tolist()
joblib.dump(feature_names, "rf_feature_names.pkl")


# %% [markdown]
# ## 4. Modeling

# %% [markdown]
# Membangun model machine learning untuk memprediksi status mahasiswa berdasarkan fitur-fitur yang telah kita bersihkan dan proses sebelumnya. Kita akan mencoba beberapa algoritma dasar sebagai baseline untuk membandingkan performa.

# %% [markdown]
# ### 4.1 Pemodelan Awal

# %% [markdown]
# Kita mulai dengan menggunakan 3 model untuk membandingkan model yang paling optimal:
# 1. **Logistic Regression**
# 2. **Random Forest**
# 3. **K-Nearest Neighbors (KNN)**

# %%
# Inisialisasi model
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier()
}

# Evaluasi tiap model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"\n📊 Model: {name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Tidak Dropout", "Dropout"]))

print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Tidak Dropout", "Dropout"]
))

# %% [markdown]
# #### 4.2 Hyperparameter Tuning – Random Forest (Binary Classification)

# %% [markdown]
# Setelah mengevaluasi model baseline, kita fokus pada penyetelan parameter model terbaik, yaitu **Random Forest**, dengan skenario klasifikasi biner (Dropout vs Tidak Dropout).  
# Tuning dilakukan untuk mencari kombinasi parameter yang menghasilkan performa optimal berdasarkan akurasi dan kemampuan mendeteksi mahasiswa dropout (recall).

# %% [markdown]
# #### Menyiapkan Parameter Grid untuk Tuning

# %%
# 4.2 Tuning Random Forest untuk Prediksi Risiko Dropout
from sklearn.model_selection import GridSearchCV

# Parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

# Inisialisasi model dasar
rf = RandomForestClassifier(random_state=42)

# Setup GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='f1',  # karena kita fokus pada deteksi kelas 1 (Dropout)
    n_jobs=-1,
    verbose=1
)

# Jalankan Grid Search
grid_search.fit(X_train, y_train)


# %% [markdown]
# ## 5. Evaluation

# %% [markdown]
# ### 5.1 Evaluasi Model Hasil Tuning
# 
# Setelah menjalankan Grid Search untuk Random Forest, kini kita evaluasi performa dari model terbaik yang ditemukan.  
# Fokus utama evaluasi:
# - Akurasi keseluruhan
# - Precision, Recall, dan F1-score khusus untuk **kelas Dropout (1)**
# - Visualisasi Confusion Matrix

# %%
# Ambil model terbaik dari tuning
best_rf = grid_search.best_estimator_

# Prediksi di data test
y_pred_best = best_rf.predict(X_test)

# Evaluasi numerik
print("📍 Best Parameters:", grid_search.best_params_)
print("\n📊 Evaluasi Model Random Forest (hasil tuning):")
print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_best, target_names=["Tidak Dropout", "Dropout"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Tidak Dropout", "Dropout"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix – Random Forest (Tuned)")
plt.show()


# %% [markdown]
# ### 5.2 Simpan Model Final & Struktur Input
# 
# Agar model dapat digunakan kembali untuk deployment, prediksi manual, atau integrasi dengan aplikasi seperti dashboard Streamlit, model hasil tuning perlu disimpan dalam format `.pkl`.  
# Selain itu, kita juga simpan daftar fitur (`feature_names`) agar input prediksi nantinya tetap konsisten dengan urutan saat training.
# 

# %%
# Simpan model hasil tuning
joblib.dump(best_rf, "best_random_forest.pkl")

# Simpan struktur urutan kolom fitur
feature_names = X_train.columns.tolist()
joblib.dump(feature_names, "rf_feature_names.pkl")

print("✅ Model dan fitur berhasil disimpan!")


# %% [markdown]
# ### 5.3 Visualisasi Proporsi Mahasiswa Berpotensi Dropout

# %%
# Ambil data mahasiswa aktif
df_enrolled = df[df["Status"] == "Enrolled"].copy()

# Pastikan urutan kolom input sesuai saat training
X_enrolled = df_enrolled[feature_names]

# Prediksi batch terhadap mahasiswa aktif
df_enrolled["Pred_Label"] = best_rf.predict(X_enrolled)

# Hitung proporsi
counts = df_enrolled["Pred_Label"].value_counts(normalize=True).sort_index()
labels = ["Tidak Dropout", "Dropout"]
colors = ["#7FB3D5", "#E74C3C"]

# Visualisasi
plt.figure(figsize=(6, 6))
plt.pie(counts, labels=labels, autopct="%.2f%%", startangle=90, colors=colors)
plt.title("Distribusi Prediksi Mahasiswa Enrolled")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5.4 Interpretasi Feature Importance – Random Forest (Binary)
# 
# Bagian ini bertujuan untuk mengidentifikasi fitur-fitur paling berpengaruh dalam model prediksi dropout.  
# Nilai _feature importance_ mencerminkan seberapa besar kontribusi relatif suatu fitur dalam menentukan hasil prediksi model Random Forest.
# 
# Dengan informasi ini, institusi pendidikan dapat:
# - Mendeteksi mahasiswa berisiko lebih awal
# - Menyusun rekomendasi intervensi yang lebih tepat sasaran
# - Menyederhanakan model ke fitur-fitur paling signifikan
# 
# Visualisasi bar chart akan menampilkan **10 fitur teratas** berdasarkan tingkat pengaruhnya dalam prediksi.
# 

# %%
# Hitung feature importance
feature_importances = best_rf.feature_importances_
features = X_train.columns

# Buat DataFrame
importance_df = pd.DataFrame({
    "Fitur": features,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

# Tampilkan Top 10 Fitur Terpenting
print("\n📊 Top 10 Fitur Terpenting dalam Prediksi Dropout:")
print(importance_df.head(10))

# Visualisasi
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Fitur"][:10][::-1], importance_df["Importance"][:10][::-1], color="darkcyan")
plt.xlabel("Pentingnya Fitur")
plt.title("Top 10 Fitur Terpenting – Random Forest (Binary Classification)")
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Interpretasi:
# 
# Berdasarkan grafik feature importance, fitur paling berpengaruh terhadap prediksi dropout adalah:
# 
# 1. **Curricular_units_2nd_sem_approved & grade**  
#    → Semakin sedikit mata kuliah yang disetujui atau nilai rendah di semester 2, semakin tinggi risiko dropout.
# 
# 2. **Curricular_units_1st_sem_approved & grade**  
#    → Semester pertama juga tetap kritikal sebagai fondasi awal akademik mahasiswa.
# 
# 3. **Tuition_fees_up_to_date**  
#    → Mahasiswa yang memiliki tunggakan cenderung memiliki risiko dropout lebih tinggi, mungkin karena tekanan finansial.
# 
# 4. **Evaluations (1st & 2nd semester)**  
#    → Banyaknya evaluasi bisa berkaitan dengan keaktifan mahasiswa atau jumlah mata kuliah yang diambil.
# 
# 5. **Admission_grade & Previous_qualification_grade**  
#    → Nilai masuk kampus dan nilai sebelumnya menunjukkan seberapa siap mahasiswa saat memulai perkuliahan.
# 
# 6. **Age_at_enrollment**  
#    → Bisa jadi mencerminkan tantangan non-akademik; misalnya mahasiswa yang lebih tua mungkin memiliki beban tambahan seperti pekerjaan atau keluarga.
# 
# Kesimpulannya, kombinasi antara **kinerja akademik awal, kondisi keuangan, dan kesiapan awal mahasiswa** adalah sinyal dropout paling kuat menurut model.
# 

# %% [markdown]
# ## 6. Kesimpulan
# 
# Model prediksi dropout yang dibangun menggunakan Random Forest dengan pendekatan klasifikasi biner berhasil mencapai performa yang sangat baik:
# 
# - **Akurasi**: 87.68%
# - **Recall untuk kelas Dropout**: 72%
# - **F1-score Dropout**: 0.79
# - **Fitur-fitur terpenting**:
#   - Nilai & jumlah MK disetujui di semester 1 & 2
#   - Status pembayaran biaya kuliah
#   - Nilai masuk dan kesiapan akademik awal
# 
# Model ini tidak hanya memberikan prediksi yang akurat, tetapi juga mengungkap **indikator utama risiko dropout**, yang bisa dijadikan dasar strategi intervensi.
# 
# Seluruh pipeline juga telah disiapkan untuk digunakan ulang, baik dalam format file `.pkl` untuk deployment maupun dataset `.csv` untuk analisis lanjutan.
# 
# > Fokus ke mahasiswa aktif (Enrolled) dan deteksi dini dropout adalah langkah konkret menuju sistem pendidikan yang lebih adaptif, suportif, dan data-driven.
# 
# 


