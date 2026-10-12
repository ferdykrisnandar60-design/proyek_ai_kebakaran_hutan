# %%
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('data/data_bersih_lengkap.csv').dropna(subset=['clean_text'])
print(df.shape)
print(df['target'].value_counts())

# %%
# Tahap 5: Train-Test Split (stratified)
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['target']
)
print("Data training:", len(train_df))
print("Data testing :", len(test_df), "| subset kebakaran:", test_df['is_fire'].sum())

# %%
# Ekstraksi fitur: vectorizer hanya belajar dari data training
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train = vectorizer.fit_transform(train_df['clean_text'])
X_test = vectorizer.transform(test_df['clean_text'])
y_train = train_df['target']
y_test = test_df['target']
print("Ukuran fitur:", X_train.shape)

# %%
# Tahap 6: Pemodelan dengan dua algoritma
model_lr = LogisticRegression(max_iter=1000, class_weight='balanced')
model_lr.fit(X_train, y_train)

model_nb = MultinomialNB()
model_nb.fit(X_train, y_train)
print("Logistic Regression dan Naive Bayes selesai dilatih!")

# %%
joblib.dump(model_lr, 'model/model_logistic_regression.pkl')
joblib.dump(model_nb, 'model/model_naive_bayes.pkl')
joblib.dump(vectorizer, 'model/vectorizer_kebakaran.pkl')
joblib.dump((X_test, y_test, test_df['is_fire'].values), 'model/test_data.pkl')
print("Model, vectorizer, dan data uji tersimpan di folder model/")
# %%
