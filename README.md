# Klasifikasi Laporan Kebakaran Hutan Berbasis NLP

Proyek mata kuliah **Kecerdasan Buatan**: sistem NLP yang mengklasifikasikan laporan kebakaran hutan sebagai **indikasi kejadian nyata** atau **bukan kejadian nyata**, sebagai upaya mendukung pelestarian ekosistem hutan. Sistem dapat menerima laporan berbahasa Indonesia melalui penerjemahan otomatis.

## Latar Belakang

Kebakaran hutan dan lahan (karhutla) menjadi ancaman kronis bagi ekosistem hutan Indonesia. Banjir laporan dari masyarakat dan media sosial, yang bercampur dengan informasi tidak relevan, menyulitkan petugas menentukan prioritas penanganan. Proyek ini membangun model klasifikasi teks untuk membantu menyaring laporan tersebut.

## Dataset

- **Sumber:** [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started) (Kaggle), berkas `train.csv`.
- **Label:** `1` = tweet tentang bencana nyata, `0` = bukan bencana nyata.
- Seluruh data digunakan tanpa data tambahan buatan sendiri. Baris yang memuat kata kunci kebakaran (`fire, wildfire, forest fire, burning, blaze, smoke`) ditandai di kolom `is_fire` untuk analisis dan evaluasi khusus topik kebakaran.
- **Catatan:** label menunjukkan kejadian nyata, bukan tingkat urgensi, sehingga urgensi hanya didekati secara proxy.
- Dataset tidak disertakan di repositori sesuai aturan Kaggle. Unduh sendiri `train.csv` dan letakkan di folder `data/`.

## Alur Pengerjaan (9 Tahap)

| No | Tahap | Berkas |
|----|-------|--------|
| 1 | Pengumpulan dataset | `data/train.csv` |
| 2 | Cleaning & preprocessing | `preprocessing.py` |
| 3 | Exploratory Data Analysis | `eda.py` |
| 4 | Feature selection (TF-IDF) | `feature_selection.py` |
| 5-6 | Train-test split & pemodelan | `modeling.py` |
| 7 | Evaluasi | `evaluation.py` |
| 8 | Prediksi & interpretasi | `prediksi.py` |
| 9 | Kesimpulan & rekomendasi | dokumen laporan |

## Metode

- **Fitur:** TF-IDF (`stop_words='english'`, `max_features=5000`), di-fit hanya pada data latih.
- **Pembagian data:** 80:20, stratified, `random_state=42`.
- **Algoritma:**
  - Logistic Regression (`class_weight='balanced'`)
  - Multinomial Naive Bayes
- **Evaluasi:** Accuracy, Precision, Recall, F1-Score, confusion matrix, dan cross-validation 5-fold.

## Hasil

| Model | Cakupan | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| Logistic Regression | Semua data uji | 0.803 | 0.780 | 0.748 | 0.764 |
| Logistic Regression | Subset kebakaran | 0.811 | 0.775 | 0.917 | 0.840 |
| Multinomial NB | Semua data uji | 0.801 | 0.829 | 0.672 | 0.742 |
| Multinomial NB | Subset kebakaran | 0.739 | 0.836 | .0.767 | 0.800 |

Ketentuan dosen: akurasi minimal 50%.

## Cara Kerja Prediksi

1. Laporan berbahasa Indonesia diterjemahkan ke bahasa Inggris (`deep-translator`, cadangan MyMemory).
2. Teks dibersihkan dengan `clean_text`, lalu diubah menjadi vektor TF-IDF.
3. Model Logistic Regression menghitung probabilitas kejadian nyata.
4. Probabilitas diterjemahkan menjadi keputusan:

| Probabilitas | Keputusan |
|--------------|-----------|
| >= 0,60 | DARURAT (indikasi kejadian nyata) |
| 0,40 - 0,60 | PERLU DITINJAU |
| <= 0,40 | BUKAN KEJADIAN NYATA / TIDAK RELEVAN |

## Struktur Proyek

```
.
├── data/                  # train.csv (unduh dari Kaggle) dan hasil preprocessing
├── model/                 # model, vectorizer, grafik (dihasilkan otomatis)
├── preprocessing.py
├── eda.py
├── feature_selection.py
├── modeling.py
├── evaluation.py
├── prediksi.py
├── requirements.txt
└── README.md
```

## Cara Menjalankan

```bash
# 1. Clone repositori
git clone <url-repositori>
cd <nama-folder>

# 2. Buat dan aktifkan virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 3. Instal dependensi
pip install -r requirements.txt

# 4. Letakkan train.csv di folder data/, lalu jalankan berurutan
python preprocessing.py
python eda.py
python feature_selection.py
python modeling.py
python evaluation.py
python prediksi.py
```

Berkas `.py` juga memakai penanda `# %%` sehingga dapat dijalankan per sel di VS Code (Interactive Window).

## Contoh Penggunaan

```python
prediksi_laporan("Terlihat api besar membakar hutan dekat desa kami")
```

## Keterbatasan

- Dataset berbahasa Inggris dan berasal dari tweet umum, belum mewakili laporan karhutla Indonesia.
- Kualitas terjemahan memengaruhi hasil prediksi.
- Layanan terjemahan gratis dapat membatasi jumlah permintaan.
- Label dataset adalah kejadian nyata, bukan tingkat urgensi.

## Pengembangan Selanjutnya

- Korpus laporan karhutla berbahasa Indonesia dengan label urgensi.
- Algoritma lain (SVM, Random Forest, IndoBERT).
- Kelas urgensi bertingkat, antarmuka web atau bot.

## Referensi

- Gusti, K. W. (2023). Klasifikasi Bencana Alam pada Twitter Menggunakan Naive Bayes, SVM dan Logistic Regression. *Technologia*, 14(4), 349-358.
- Greenpeace Indonesia. (2024). *Bagai Api dalam Sekam: Data Karhutla Kronis 2023*.
- Kaggle. Natural Language Processing with Disaster Tweets.

## Penulis

Ferdy Krisnandar