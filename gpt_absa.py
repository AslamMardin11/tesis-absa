import streamlit as st
import re
import numpy as np
import pandas as pd
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# === Preprocessing ===
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
stemmer = StemmerFactory().create_stemmer()

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = text.lower().strip()
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    text = re.sub(r"\s+", " ", text)
    return text

def load_lexicon(filepath):
    for sep in [',', ';', '\t']:
        try:
            df = pd.read_csv(filepath, sep=sep, header=None, names=["word", "weight"])
            df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
            return dict(zip(df['word'].str.lower().str.strip(), df['weight'].dropna()))
        except:
            continue
    return {}

def get_lexicon_score(text, lexicon):
    words = text.split()
    return sum(lexicon.get(w, 0) for w in words)

def predict_sentiment(text, tokenizer, model, max_len, lexicon):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=max_len)
    proba = model.predict(pad, verbose=0)[0]
    label = np.argmax(proba)
    labels = ["Negatif", "Netral", "Positif"]
    confidence = round(proba[label] * 100, 2)
    lex_score = get_lexicon_score(cleaned, lexicon)
    final_score = round(0.7 * proba[label] + 0.3 * (1 if lex_score > 0 else -1 if lex_score < 0 else 0), 2)
    return labels[label], confidence, lex_score, final_score

# === Load Model & Tokenizer ===
model = load_model("rnn_model.h5")

with open("tokenizer.json") as f:
    tokenizer = tokenizer_from_json(json.load(f))

max_len = 100
lexicon_pos = load_lexicon("lexicon_positive.csv")
lexicon_neg = load_lexicon("lexicon_negative.csv")
lexicon = {**lexicon_pos, **lexicon_neg}

# === Streamlit UI ===
st.set_page_config(page_title="Sentiment Ponpes", layout="wide")
st.title("📊 Aspect-Based Sentiment Analysis untuk Kinerja Ponpes")

text_input = st.text_area("Masukkan komentar/ulasan:")

if st.button("Analisis"):
    if not text_input.strip():
        st.warning("Mohon isi ulasan terlebih dahulu.")
    else:
        label, confidence, lex_score, final_score = predict_sentiment(text_input, tokenizer, model, max_len, lexicon)
        st.success(f"Hasil Sentimen: **{label}** ({confidence}%)")
        st.write(f"Skor Lexicon: `{lex_score}`")
        st.write(f"Skor Gabungan: `{final_score}`")
