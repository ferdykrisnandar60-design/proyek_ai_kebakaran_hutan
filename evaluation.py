# %%
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

model_lr = joblib.load('model/model_logistic_regression.pkl')
model_nb = joblib.load('model/model_naive_bayes.pkl')
X_test, y_test, is_fire = joblib.load('model/test_data.pkl')
y_test = np.asarray(y_test)
is_fire = np.asarray(is_fire)
print("Data uji:", len(y_test), "| subset kebakaran:", int(is_fire.sum()))

# %%
# Tabel perbandingan: dua model x dua cakupan data uji
baris = []
semua = np.ones(len(y_test), dtype=bool)
for nama, model in [('Logistic Regression', model_lr), ('Naive Bayes', model_nb)]:
    y_pred = model.predict(X_test)
    for cakupan, m in [('Semua data uji', semua), ('Subset kebakaran', is_fire)]:
        baris.append({
            'Model': nama,
            'Cakupan': cakupan,
            'Jumlah': int(m.sum()),
            'Accuracy': accuracy_score(y_test[m], y_pred[m]),
            'Precision': precision_score(y_test[m], y_pred[m]),
            'Recall': recall_score(y_test[m], y_pred[m]),
            'F1-Score': f1_score(y_test[m], y_pred[m]),
        })

df_hasil = pd.DataFrame(baris)
print(df_hasil.round(3).to_string(index=False))

# %%
# Confusion matrix
for nama, model in [('Logistic Regression', model_lr), ('Naive Bayes', model_nb)]:
    y_pred = model.predict(X_test)
    print(f"Confusion Matrix {nama} (semua data uji):")
    print(confusion_matrix(y_test, y_pred))
    print(f"Confusion Matrix {nama} (subset kebakaran):")
    print(confusion_matrix(y_test[is_fire], y_pred[is_fire]))
    print("Format: [[TN FP] [FN TP]]\n")

# %%
# Grafik perbandingan
metrik = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, cakupan in zip(axes, ['Semua data uji', 'Subset kebakaran']):
    sub = df_hasil[df_hasil['Cakupan'] == cakupan]
    x = np.arange(len(metrik))
    lebar = 0.35
    for i, (_, r) in enumerate(sub.iterrows()):
        ax.bar(x + (i - 0.5) * lebar, [r[m] for m in metrik], lebar, label=r['Model'])
    ax.set_xticks(x)
    ax.set_xticklabels(metrik)
    ax.set_ylim(0, 1)
    ax.set_title(cakupan)
    ax.legend()
plt.tight_layout()
plt.savefig('model/perbandingan_model.png')
plt.show()

# %%
# Cross-validation 5 lipatan (menguji kestabilan hasil)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

df = pd.read_csv('data/data_bersih_lengkap.csv').dropna(subset=['clean_text'])
pipeline = make_pipeline(
    TfidfVectorizer(stop_words='english', max_features=5000),
    LogisticRegression(max_iter=1000, class_weight='balanced')
)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for metrik_cv in ['accuracy', 'precision', 'recall', 'f1']:
    skor = cross_val_score(pipeline, df['clean_text'], df['target'], cv=skf, scoring=metrik_cv)
    print(f"{metrik_cv:10s}: rata-rata {skor.mean():.3f} (+/- {skor.std():.3f})")
# %%
