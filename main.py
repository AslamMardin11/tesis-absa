import pandas as pd
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Load dataset Excel (ganti 'dataset.xls' dengan path file asli)
data = pd.read_excel('2000_Kalimat_Kinerja_Pesantren_Unik.xlsx')

# Load lexicon positif dan negatif (ganti path sesuai file)
positive_words = pd.read_csv('positive.tsv', sep='\t', header=None, names=['word'])
negative_words = pd.read_csv('negative.tsv', sep='\t', header=None, names=['word'])

positive_words = set(positive_words['word'].str.strip().str.lower())
negative_words = set(negative_words['word'].str.strip().str.lower())

# Inisialisasi stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|@\w+|#[A-Za-z0-9_]+", "", text)  # hapus URL, mention, hashtag
    text = re.sub(f"[{string.punctuation}]", " ", text)       # hapus tanda baca
    text = stemmer.stem(text)
    text = re.sub(r"\s+", " ", text).strip()                 # hilangkan spasi berlebih
    return text

data['Clean'] = data['Kalimat'].apply(preprocess)

def get_sentiment(text):
    pos, neg = 0, 0
    for word in text.split():
        if word in positive_words:
            pos += 1
        elif word in negative_words:
            neg += 1
    if pos > neg:
        return 2  # Positif
    elif neg > pos:
        return 0  # Negatif
    else:
        return 1  # Netral

data['Sentimen'] = data['Clean'].apply(get_sentiment)

# Cek distribusi kelas sentimen
print("Distribusi kelas sentimen:")
print(data['Sentimen'].value_counts())

# Jika hanya ada satu kelas, beri peringatan
if data['Sentimen'].nunique() == 1:
    print("Peringatan: Dataset hanya memiliki satu kelas sentimen, sebaiknya data lexicon atau data input diperiksa ulang.")

# Contoh print beberapa baris data hasil preprocessing dan sentimen
print(data[['Kalimat', 'Clean', 'Sentimen']].head())
