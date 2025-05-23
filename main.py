# Import Library
# !pip install pandas numpy gensim tensorflow scikit-learn Sastrawi matplotlib

import pandas as pd
import numpy as np
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

print("berhasil mengimport library")

# Load dataset
data = pd.read_excel("2000_Kalimat_Kinerja_Pesantren_Unik.xlsx")
print("Contoh 5 data pertama:")
print(data.head())
print("\nDistribusi Sentimen:")
print(data["Sentimen"].value_counts())

# text processing
# Fungsi cleaning teks
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)  # Hapus URL, mention, hashtag
    text = re.sub(r'[^\w\s]', '', text)  # Hapus tanda baca
    text = text.lower()  # Case folding
    
    # Stopword removal
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    text = stopword.remove(text)
    
    # Stemming
    stemmer = StemmerFactory().create_stemmer()
    text = stemmer.stem(text)
    return text

# Apply cleaning
data["Cleaned_Text"] = data["Kalimat"].apply(clean_text)
print("\nteks setelah cleaning:")
print(data[["Kalimat", "Cleaned_Text"]].head())

# pembagian train data dan testing data

# Encoding sentimen (Negatif:0, Netral:1, Positif:2)
sentimen_map = {"Negatif": 0, "Netral": 1, "Positif": 2}
data["Sentimen_Encoded"] = data["Sentimen"].map(sentimen_map)

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    data["Cleaned_Text"], 
    data["Sentimen_Encoded"], 
    test_size=0.2, 
    stratify=data["Sentimen_Encoded"],
    random_state=42
)
print("\nJumlah data training dan testing:")
print(f"Training: {len(X_train)}, Testing: {len(X_test)}")
print("Distribusi label training:")
print(y_train.value_counts())

# oversampling pada vector
# Gabungkan X_train dan y_train
train_data = pd.DataFrame({"text": X_train, "label": y_train})
# Pisahkan per kelas
df_netral = train_data[train_data["label"] == 1]
df_positif = train_data[train_data["label"] == 2]
df_negatif = train_data[train_data["label"] == 0]

# Oversampling positif & negatif
df_positif_oversampled = resample(df_positif, replace=True, n_samples=len(df_netral), random_state=42)
df_negatif_oversampled = resample(df_negatif, replace=True, n_samples=len(df_netral), random_state=42)

# print(df_positif_oversampled)
# Gabungkan kembali
train_data_balanced = pd.concat([df_netral, df_positif_oversampled, df_negatif_oversampled])
print("\nDistribusi label setelah oversampling:")
print(train_data_balanced["label"].value_counts())

# tokenizing dan pading
# Tokenisasi
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_data_balanced["text"])

# Konversi teks ke sequence
X_train_seq = tokenizer.texts_to_sequences(train_data_balanced["text"])
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding
max_len = 100  # Panjang maksimal sequence
X_train_padded = pad_sequences(X_train_seq, maxlen=max_len)
X_test_padded = pad_sequences(X_test_seq, maxlen=max_len)

print("\nContoh hasil padding:")
print(X_train_padded[0])

# Hyperparameter optimal dari pengujian
embed_dim = 100
lstm_units = 198
dropout_rate = 0.2

# Model architecture
model = Sequential([
    Embedding(input_dim=5000, output_dim=embed_dim, input_length=max_len),
    SpatialDropout1D(dropout_rate),
    LSTM(lstm_units, dropout=dropout_rate),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nSummary model:")
model.summary()

# Pelatihan Model   
# Callback untuk early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3)

# Training
history = model.fit(
    X_train_padded, 
    train_data_balanced["label"],
    epochs=10,
    batch_size=32,
    validation_data=(X_test_padded, y_test),
    callbacks=[early_stop]
)

# Plot akurasi
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()


# Evaluasi Model
# Prediksi
y_pred = np.argmax(model.predict(X_test_padded), axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=sentimen_map.keys()))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# simpan model
model.save("model_lstm_pesantren.h5")


from tensorflow.keras.models import load_model
import numpy as np

# Load model
model = load_model("model_lstm_pesantren.h5")
def predict_aspect(text):
    # Rule-based: Cek kata kunci untuk setiap aspek
    aspect_keywords = {
        "Kualitas Pengajaran": ["ajar", "ustadz", "kurikulum", "belajar"],
        "Fasilitas": ["asrama", "kamar", "fasilitas", "bersih", "kotor"],
        "Kualitas Guru": ["guru", "ustadz", "pengajar", "sabar"],
        "Prestasi": ["prestasi", "juara", "lomba", "menang"]
    }
    
    cleaned_text = clean_text(text)
    words = cleaned_text.split()
    
    # Cek kata yang cocok dengan aspek
    for aspect, keywords in aspect_keywords.items():
        if any(keyword in words for keyword in keywords):
            return aspect
    return "Umum"  # Default jika tidak ada aspek terdeteksi

def predict_text_with_aspect(text):
    sentiment = predict_sentiment(text)
    aspect = predict_aspect(text)
    return {
        "Text": text,
        "Sentimen": sentiment,
        "Aspek": aspect
    }

# Contoh penggunaan
new_texts = [
    "Fasilitas pesantren tidak nyaman dan bersih.",
    "Pengajaran ustadz kurang jelas.",
    "Prestasi santri di olimpiade membanggakan."
]

# Hasil prediksi
results = [predict_text_with_aspect(text) for text in new_texts]

# Konversi ke DataFrame untuk tampilan rapi
import pandas as pd
results_df = pd.DataFrame(results)
print("\nHasil Prediksi:")
print(results_df)