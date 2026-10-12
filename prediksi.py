# %%
import re
import time
import joblib
from deep_translator import GoogleTranslator, MyMemoryTranslator

model = joblib.load('model/model_logistic_regression.pkl')
vectorizer = joblib.load('model/vectorizer_kebakaran.pkl')
print("Model dan vectorizer berhasil dimuat")

# %%
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# %%
def terjemahkan(teks, percobaan=3):
    for i in range(percobaan):
        try:
            return GoogleTranslator(source='id', target='en').translate(teks)
        except Exception:
            time.sleep(2 * (i + 1))
    return MyMemoryTranslator(source='indonesian', target='english').translate(teks)

# %%
BATAS_DARURAT = 0.60      # probabilitas darurat >= batas ini -> DARURAT
BATAS_TIDAK_DARURAT = 0.40  # probabilitas darurat <= batas ini -> BUKAN KEJADIAN NYATA

def prediksi_laporan(teks_indonesia):
    teks_inggris = terjemahkan(teks_indonesia)
    teks_bersih = clean_text(teks_inggris)
    vektor = vectorizer.transform([teks_bersih])
    p_darurat = model.predict_proba(vektor)[0][1]

    if p_darurat >= BATAS_DARURAT:
        label = "DARURAT (indikasi kejadian nyata)"
        rekomendasi = "Segera hubungi BPBD/petugas pemadam kebakaran setempat."
    elif p_darurat <= BATAS_TIDAK_DARURAT:
        label = "BUKAN KEJADIAN NYATA / TIDAK RELEVAN"
        rekomendasi = "Laporan dicatat, tidak memerlukan respons segera."
    else:
        label = "PERLU DITINJAU"
        rekomendasi = "Keyakinan model rendah, mohon tinjau manual."

    return {
        'teks_asli': teks_indonesia,
        'teks_terjemahan': teks_inggris,
        'label': label,
        'prob_darurat': f"{p_darurat*100:.1f}%",
        'rekomendasi': rekomendasi,
    }

# %%
contoh_laporan = [
    "Ada api besar dekat pemukiman warga, asap tebal sekali",
    "Kebakaran hutan sudah meluas, warga diminta mengungsi",
    "Saya ingin tahu jadwal penyuluhan tentang pencegahan kebakaran hutan",
    "Semangat saya sedang membara hari ini",
    "Konsernya panas sekali, penontonnya heboh",
]

for laporan in contoh_laporan:
    hasil = prediksi_laporan(laporan)
    print(f"Laporan        : {hasil['teks_asli']}")
    print(f"Terjemahan     : {hasil['teks_terjemahan']}")
    print(f"Label          : {hasil['label']}")
    print(f"Prob. darurat  : {hasil['prob_darurat']}")
    print(f"Tindakan       : {hasil['rekomendasi']}")
    print("-" * 60)
    time.sleep(1)
# %%
