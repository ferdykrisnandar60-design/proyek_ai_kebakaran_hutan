# %%
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

df = pd.read_csv('data/data_bersih_kebakaran.csv')
print(df.shape)
df.head()

# %%
# 1. Distribusi Label (Darurat vs Tidak Darurat)
print(df['target'].value_counts())
print()
print(df['target'].value_counts(normalize=True) * 100)

df['target'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribusi Label: Darurat (1) vs Tidak Darurat (0)')
plt.xlabel('Label')
plt.ylabel('Jumlah')
plt.xticks(rotation=0)
plt.savefig('model/distribusi_label.png')
plt.show()

# %%
# 2. Distribusi Panjang Teks (jumlah kata)
df['jumlah_kata'] = df['clean_text'].apply(lambda x: len(str(x).split()))

print(df['jumlah_kata'].describe())
df['jumlah_kata'].plot(kind='hist', bins=20, color='lightgreen', edgecolor='black')
plt.title('Distribusi Jumlah Kata per Laporan')
plt.xlabel('Jumlah Kata')
plt.ylabel('Frekuensi')
plt.savefig('model/distribusi_panjang_teks.png')
plt.show()

# %%
# 3. Kata yang Paling Sering Muncul - Label Darurat (1)
teks_darurat = ' '.join(df[df['target'] == 1]['clean_text'])
kata_darurat = teks_darurat.split()
top_darurat = Counter(kata_darurat).most_common(15)

print("15 Kata Paling Sering Muncul di Laporan DARURAT:")
for kata, jumlah in top_darurat:
    print(f"{kata:15s} : {jumlah}")

# %%
# 4. Kata yang Paling Sering Muncul - Label Tidak Darurat (0)
teks_tidak_darurat = ' '.join(df[df['target'] == 0]['clean_text'])
kata_tidak_darurat = teks_tidak_darurat.split()
top_tidak_darurat = Counter(kata_tidak_darurat).most_common(15)

print("15 Kata Paling Sering Muncul di Laporan TIDAK DARURAT:")
for kata, jumlah in top_tidak_darurat:
    print(f"{kata:15s} : {jumlah}")

# %%
# 5. Bandingkan panjang teks rata-rata per label
print(df.groupby('target')['jumlah_kata'].mean())

df.boxplot(column='jumlah_kata', by='target')
plt.title('Perbandingan Panjang Teks per Label')
plt.suptitle('')
plt.xlabel('Label (0=Tidak Darurat, 1=Darurat)')
plt.ylabel('Jumlah Kata')
plt.savefig('model/boxplot_panjang_teks.png')
plt.show()
