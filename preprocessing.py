# %%
import pandas as pd
import re

df = pd.read_csv('data/train.csv')
print(df.shape)
df.head()

# %%
keywords_fire = ['fire', 'wildfire', 'forest fire', 'burning', 'blaze', 'smoke']
df_fire = df[df['text'].str.contains('|'.join(keywords_fire), case=False, na=False)]

print(f"Total data setelah difilter: {len(df_fire)}")
df_fire.head()

# %%
print("Missing value:\n", df_fire.isnull().sum())

df_fire = df_fire.dropna(subset=['text', 'target'])
df_fire = df_fire.drop_duplicates(subset=['text'])

print("Jumlah data setelah dibersihkan:", len(df_fire))

# %%
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df_fire['clean_text'] = df_fire['text'].apply(clean_text)
df_fire[['text', 'clean_text']].head(10)

# %%
print(df_fire['target'].value_counts())

# %%
df_fire[['text', 'clean_text', 'target']].to_csv('data/data_bersih_kebakaran.csv', index=False)
print("Tersimpan sebagai data/data_bersih_kebakaran.csv")