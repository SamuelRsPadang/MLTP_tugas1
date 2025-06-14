# Proyek Predictive Analytics : Prediksi Risiko Stroke

## Domain Proyek

Stroke merupakan salah satu penyebab utama kematian dan kecacatan di dunia. Menurut data dari World Health Organization (WHO), sekitar 15 juta orang di dunia mengalami stroke setiap tahun. Dari jumlah tersebut, sekitar 5 juta orang meninggal dan 5 juta lainnya menjadi cacat permanen akibat komplikasi yang ditimbulkan \[WHO, 2023].

Stroke dapat dicegah apabila faktor risikonya seperti tekanan darah tinggi, diabetes, merokok, obesitas, dan gaya hidup tidak sehat dapat dideteksi dan dikendalikan sedini mungkin. Namun, deteksi dini sering kali terhambat karena keterbatasan sumber daya medis, terutama di negara berkembang.

Dengan berkembangnya teknologi dan ketersediaan data kesehatan, pendekatan berbasis machine learning dapat membantu memprediksi risiko stroke secara otomatis dan cepat. Model prediktif ini dapat digunakan oleh tenaga medis atau aplikasi kesehatan digital untuk memberi peringatan dini bagi pasien yang berisiko tinggi.

### Mengapa dan Bagaimana Masalah Ini Harus Diselesaikan

Masalah ini penting untuk diselesaikan karena:

* Stroke bisa dicegah jika faktor risikonya dikenali sejak dini.
* Sistem prediksi berbasis machine learning dapat membantu dalam skrining awal, terutama di daerah dengan akses terbatas ke tenaga medis.
* Efisiensi waktu dan biaya: deteksi awal melalui data digital mengurangi kebutuhan uji klinis yang mahal dan memakan waktu.

Dengan membangun model prediksi stroke, instansi kesehatan, rumah sakit, maupun aplikasi kesehatan personal dapat memberikan peringatan atau rekomendasi medis yang lebih tepat dan personal.

### Referensi

* World Health Organization. (2023). Stroke: Key Facts. Diakses dari: [https://www.who.int/news-room/fact-sheets/detail/stroke](https://www.who.int/news-room/fact-sheets/detail/stroke)
* Benjamin, E. J., et al. (2019). Heart disease and stroke statistics—2019 update: a report from the American Heart Association. Circulation, 139(10), e56–e528. [https://doi.org/10.1161/CIR.0000000000000659](https://doi.org/10.1161/CIR.0000000000000659)
* Feigin, V. L., et al. (2021). Global, regional, and national burden of stroke and its risk factors, 1990–2019: a systematic analysis for the Global Burden of Disease Study 2019. The Lancet Neurology, 20(10), 795–80.

## Business Understanding

### Problem Statement

Stroke merupakan penyakit yang dapat menyebabkan kematian atau kecacatan jangka panjang. Banyak pasien stroke datang ke rumah sakit setelah gejala muncul, padahal stroke dapat dicegah jika faktor risikonya terdeteksi lebih awal. Masalah utamanya adalah:

* Bagaimana cara mengidentifikasi individu yang berisiko mengalami stroke berdasarkan data profil kesehatan mereka?
* Dapatkah kita membangun model machine learning yang mampu memprediksi risiko stroke secara akurat dan efisien?

### Goals

Tujuan dari proyek ini adalah:

* Mengembangkan model machine learning untuk memprediksi risiko stroke berdasarkan fitur-fitur seperti usia, tekanan darah, kadar glukosa, riwayat hipertensi, dan kebiasaan merokok.
* Memberikan solusi prediktif yang bisa diintegrasikan dalam sistem pendukung keputusan medis (Clinical Decision Support System), atau aplikasi kesehatan digital untuk masyarakat umum.
* Membantu pencegahan stroke dini melalui identifikasi kelompok berisiko tinggi secara otomatis.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah Stroke Prediction Dataset, tersedia di: [https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset). Dataset ini dirancang untuk memprediksi kemungkinan seseorang mengalami stroke berdasarkan berbagai faktor risiko klinis dan demografis.

### Ringkasan Dataset

* Jumlah Baris: 5.110
* Jumlah Kolom: 12
* Variabel Target: stroke (biner: 0 = tidak stroke, 1 = stroke)
* Tipe Data:

  * Numerik: id (integer), age (float), avg\_glucose\_level (float), bmi (float)
  * Kategorikal: gender (object), ever\_married (object), work\_type (object), Residence\_type (object), smoking\_status (object)
  * Biner: hypertension (integer: 0 atau 1), heart\_disease (integer: 0 atau 1), stroke (integer: 0 atau 1)

### Deskripsi Fitur

Dataset ini memiliki 12 kolom berikut:

1. id: Pengenal unik untuk setiap individu (integer).
2. gender: Jenis kelamin individu, dengan kategori 'Male', 'Female', atau 'Other' (object).
3. age: Usia individu dalam tahun (float, berkisar dari 0,08 hingga 82).
4. hypertension: Menunjukkan apakah individu memiliki hipertensi (0 = tidak, 1 = ya, integer).
5. heart\_disease: Menunjukkan apakah individu memiliki penyakit jantung (0 = tidak, 1 = ya, integer).
6. ever\_married: Status pernikahan, dengan kategori 'Yes' atau 'No' (object).
7. work\_type: Jenis pekerjaan, dengan kategori 'children', 'Govt\_job', 'Never\_worked', 'Private', dan 'Self-employed' (object).
8. Residence\_type: Tempat tinggal, dengan kategori 'Urban' atau 'Rural' (object).
9. avg\_glucose\_level: Rata-rata kadar glukosa darah individu (float, berkisar dari 55,12 hingga 271,74).
10. bmi: Indeks Massa Tubuh (Body Mass Index), menunjukkan proporsi berat dan tinggi badan (float, berkisar dari 10,3 hingga 97,6).
11. smoking\_status: Kebiasaan merokok, dengan kategori 'formerly smoked', 'never smoked', 'smokes', atau 'Unknown' (object).
12. stroke: Variabel target yang menunjukkan apakah individu mengalami stroke (0 = tidak, 1 = ya, integer).

### Kondisi Data Awal

* Nilai yang Hilang: Dataset memiliki nilai yang hilang pada kolom bmi, dengan 201 entri kosong (sekitar 3,9% dari total 5.110 baris). Tidak ada kolom lain yang memiliki nilai yang hilang.
* Duplikasi: Tidak ada baris duplikat dalam dataset, sebagaimana dikonfirmasi dengan pemeriksaan df.duplicated().sum().
* Ketidakseimbangan Kelas: Variabel target stroke sangat tidak seimbang, dengan hanya 249 kasus stroke (1) dibandingkan 4.861 kasus tanpa stroke (0), mewakili sekitar 4,9% kasus positif.

### Sumber Dataset

Dataset ini tersedia secara publik di Kaggle: Stroke Prediction Dataset . Dataset ini cocok untuk mengembangkan model machine learning untuk prediksi kesehatan, khususnya untuk deteksi dini risiko stroke.

## Data Preparation

1. Penanganan Nilai yang Hilang:

* Kolom bmi memiliki 201 nilai yang hilang (3,9% dari data). Nilai yang hilang diisi dengan median kolom bmi.

2. Pemeriksaan Duplikasi:

* Dataset diperiksa dan tidak ada baris duplikat.

3. Encoding Kolom Kategorikal Biner:

* Kolom gender, ever\_married, Residence\_type diubah menjadi nilai numerik dengan Label Encoding.

4. Encoding Kolom Multikategori:

* Kolom work\_type dan smoking\_status diubah menjadi fitur biner dengan One-Hot Encoding.

5. Penghapusan Kolom Tidak Relevan:

* Kolom id dihapus. Kolom stroke dipisahkan sebagai variabel target.

6. Scaling Fitur Numerik:

* Fitur age, avg\_glucose\_level, bmi diskalakan menggunakan StandardScaler.

7. Pembagian Data:

* Dataset dibagi menjadi training set (80%) dan test set (20%) dengan stratifikasi.

## Modeling

Model yang digunakan:

1. Logistic Regression:

* class\_weight='balanced'
* Recall tinggi (0.80), cocok untuk deteksi awal meskipun precision rendah.

2. Random Forest Classifier:

* n\_estimators=100
* Tidak mendeteksi kelas stroke sama sekali (recall=0).

3. XGBoost Classifier:

* scale\_pos\_weight=19
* Precision dan ROC-AUC cukup baik (0.19 dan 0.79).

4. K-Nearest Neighbors:

* n\_neighbors=5
* Tidak mendeteksi kelas stroke.

## Evaluation

| Model               | Accuracy | Precision (1) | Recall (1) | F1 (1) | ROC-AUC |
| ------------------- | -------- | ------------- | ---------- | ------ | ------- |
| Logistic Regression | 0.75     | 0.14          | 0.80       | 0.24   | 0.84    |
| Random Forest       | 0.95     | 0.00          | 0.00       | 0.00   | 0.78    |
| XGBoost             | 0.94     | 0.19          | 0.10       | 0.13   | 0.79    |
| KNN                 | 0.94     | 0.00          | 0.00       | 0.00   | 0.60    |

Model terbaik: **XGBoost**, karena:

* Precision lebih tinggi dari Logistic Regression.
* Recall masih lebih baik dibanding Random Forest dan KNN.
* ROC-AUC kompetitif (0.79).

## Kesimpulan

* Ketidakseimbangan data menjadi tantangan utama.
* Logistic Regression efektif untuk skrining awal.
* XGBoost memberikan keseimbangan terbaik antara deteksi kelas minoritas dan pengendalian false positive.
* Perlu eksplorasi lebih lanjut dengan SMOTE atau tuning hyperparameter XGBoost.
