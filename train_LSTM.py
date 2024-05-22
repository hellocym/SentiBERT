import string
import numpy as np 
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Flatten, Dense, SpatialDropout1D, Bidirectional, LSTM, BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from alive_progress import alive_bar
import os

# 设置环境变量以确保只使用特定的 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# Load and preprocess data
DATASET_ENCODING = 'ISO-8859-1'
data = pd.read_csv('./data/training.1600000.processed.noemoticon.csv', encoding=DATASET_ENCODING, header=None)
data = data[[5, 0]]
data.columns = ['tweet', 'sentiment']
data['sentiment'] = data['sentiment'].replace(4, 1)

stop_words = set(stopwords.words('english'))

def preprocess(text):
    # Removing URLs
    text = re.sub(r"https?://\S+|www\.\S+"," ",text)
    # Removing html tags
    text = re.sub(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});"," ",text)
    # Removing the Punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Removing words that have numbers 
    text = re.sub(r"\w*\d\w*", " ", text)
    # Removing Digits 
    text = re.sub(r"[0-9]+", " ", text)
    # Cleaning white spaces
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()
    # Check stop words
    tokens = []
    for token in text.split():
        if token not in stop_words and len(token) > 3:
            tokens.append(token)
    return " ".join(tokens)

preprocessed_file = 'preprocessed_and_padded_data.npz'

if not os.path.exists(preprocessed_file):
    print("Preprocessing data...")
    with alive_bar(len(data), title="Preprocessing") as bar:
        preprocessed_tweets = []
        for tweet in data['tweet']:
            preprocessed_tweets.append(preprocess(tweet))
            bar()
    data['tweet'] = preprocessed_tweets

    X = data['tweet']
    y = data['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=7)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=2./9, random_state=7)

    print("Train Data size:", len(X_train), len(y_train))
    print("Validation Data size:", len(X_val), len(y_val))
    print("Test Data size", len(X_test), len(y_test))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    # Convert text to sequences of integers
    print("Tokenizing and padding sequences...")
    with alive_bar(3, title="Tokenizing and Padding") as bar:
        X_train = tokenizer.texts_to_sequences(X_train)
        bar()
        X_val = tokenizer.texts_to_sequences(X_val)
        bar()
        X_test = tokenizer.texts_to_sequences(X_test)
        bar()

    max_length = max([len(seq) for seq in X_train])
    X_train = pad_sequences(X_train, maxlen=max_length)
    X_val = pad_sequences(X_val, maxlen=max_length)
    X_test = pad_sequences(X_test, maxlen=max_length)
    print(f"After padding: {X_train.shape}")
    print(f"After padding: {X_val.shape}")
    print(f"After padding: {X_test.shape}")

    np.savez_compressed(preprocessed_file, 
                        X_train=X_train, y_train=y_train, 
                        X_val=X_val, y_val=y_val, 
                        X_test=X_test, y_test=y_test, 
                        word_index=np.array(tokenizer.word_index), 
                        max_length=max_length)
else:
    print("Loading preprocessed and padded data...")
    data = np.load(preprocessed_file, allow_pickle=True)
    X_train = data['X_train'][:25000]
    y_train = data['y_train'][:2500]
    X_val = data['X_val'][:25000]
    y_val = data['y_val'][:25000]
    X_test = data['X_test'][:25000]
    y_test = data['y_test'][:25000]
    word_index = data['word_index'].item()
    max_length = data['max_length']

print("Train Data size:", len(X_train), len(y_train))
print("Validation Data size:", len(X_val), len(y_val))
print("Test Data size", len(X_test), len(y_test))

vocab_size = len(word_index) + 1
embedding_dim = 128


# Define LSTM model
def create_lstm_model():
    LSTM_model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        # SpatialDropout1D(0.3),
        Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)),
        Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
        Dropout(0.5),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    LSTM_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return LSTM_model

with tf.device('/gpu:1'):
    lstm_model = create_lstm_model()

lstm_model_path = 'lstm_model.h5'

# Load model if it exists
initial_epoch_cnn = 0
initial_epoch_lstm = 0

if os.path.exists(lstm_model_path):
    with tf.device('/gpu:1'):
        lstm_model = load_model(lstm_model_path)
        print("Loaded LSTM model from disk.")
        # Extract the initial epoch from the model file name
        initial_epoch_lstm = int(lstm_model_path.split('_')[-1].split('.')[0])

# Callbacks to save the model
lstm_checkpoint = ModelCheckpoint('lstm_model_{epoch:02d}.h5', monitor='val_accuracy', save_best_only=True, mode='max', save_weights_only=False)
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)

# Training function
def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, checkpoint, initial_epoch, early_stopping, reduce_lr):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), 
              callbacks=[checkpoint, early_stopping, reduce_lr], initial_epoch=initial_epoch)

batch_size = 512
epochs = 20


with tf.device('/gpu:1'):
    train_model(lstm_model, X_train, y_train, X_val, y_val, epochs, batch_size, lstm_checkpoint, initial_epoch_lstm, early_stopping, reduce_lr)
