# %%
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('data/data_bersih_kebakaran.csv')
print(df.shape)

# %%
# 1. Bangun TF-IDF dengan seluruh kata dulu (tanpa batasan) untuk melihat total kosakata
vectorizer_full = TfidfVectorizer(stop_words='english')
X_full = vectorizer_full.fit_transform(df['clean_text'])

print(f"Total kata untuk (kosakata) sebelum dibatasi: {X_full.shape[1]}")

# %%
# 2. Bandingkan beberapa nilai max_features untuk menentukan yang optimal
for n in [500, 1000, 2000, 3000, X_full.shape[1]]:
    vec = TfidfVectorizer(stop_words='english', max_features=n)
    X = vec.fit_transform(df['clean_text'])
    print(f"max_features={n:<6} -> ukuran fitur: {X.shape}")

# %%
# 3. Lihat kata-kata dengan skor TF-IDF rata-rata tertinggi (paling berpengaruh)
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X = vectorizer.fit_transform(df['clean_text'])

skor_rata2 = np.asarray(X.mean(axis=0)).flatten()
nama_fitur = vectorizer.get_feature_names_out()

df_skor = pd.DataFrame({'kata': nama_fitur, 'skor_tfidf': skor_rata2})
df_skor = df_skor.sort_values(by='skor_tfidf', ascending=False)

print("20 Kata dengan Skor TF-IDF Rata-rata Tertinggi:")
print(df_skor.head(20).to_string(index=False))

# %%
# 4. Bandingkan kata berpengaruh di masing-masing kelas (Darurat vs Tidak Darurat)
X_darurat = vectorizer.transform(df[df['target'] == 1]['clean_text'])
X_tidak_darurat = vectorizer.transform(df[df['target'] == 0]['clean_text'])

skor_darurat = np.asarray(X_darurat.mean(axis=0)).flatten()
skor_tidak_darurat = np.asarray(X_tidak_darurat.mean(axis=0)).flatten()

df_kelas = pd.DataFrame({
    'kata': nama_fitur,
    'skor_darurat': skor_darurat,
    'skor_tidak_darurat': skor_tidak_darurat
})
df_kelas['selisih'] = df_kelas['skor_darurat'] - df_kelas['skor_tidak_darurat']

print("10 Kata Paling Kuat Menandakan DARURAT:")
print(df_kelas.sort_values('selisih', ascending=False).head(10)[['kata', 'selisih']].to_string(index=False))

print("\n10 Kata Paling Kuat Menandakan TIDAK DARURAT:")
print(df_kelas.sort_values('selisih', ascending=True).head(10)[['kata', 'selisih']].to_string(index=False))

# %%
# 5. Kesimpulan: fitur final yang dipakai
print(f"Fitur final yang dipilih: TF-IDF dengan max_features=3000")
print(f"Ukuran fitur akhir: {X.shape}")
# %%