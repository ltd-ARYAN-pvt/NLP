import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import config

negative_words = {"no","don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "mightn", 'didn', 'ain', "cannot", "isn't", "aren't", 'needn', "wasn't", "weren't", "haven't", "hasn't", "hadn't", "mustn't", "needn't", "shan't", "shouldn't",'not', "mightn't", "couldn't",'no', "hasn't"}
stop_words -= negative_words

def preprocess_text_pipeline_with_stopword_removal(text):
    #--> 1. Lowercasing
    text = text.lower()

    #--> 2. Remove HTML tags and special characters
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    #--> 3. Tokenize the text
    tokens = word_tokenize(text)
    # print(tokens)

    #--> 4. Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    # print(tokens)

    #--> 5. Lemmatize the tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # print(tokens)

    #--> 7. Join the tokens back into a string
    text = ' '.join(tokens)

    return text


def get_texts_labels(data,text_col,label_col):
    texts = data[text_col].tolist()
    labels = data[label_col].tolist()

    return config.filter_labels(texts,labels,desired_labels=config.desired_dialogue_acts)

def load_glove_embeddings(word_index,file_path=config.glove_path, embedding_dim=50):
    embeddings_index = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix

def preprocess_text_pipeline(texts, labels, num_words=15000, max_len=50):
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)
    text_sequences = tokenizer.texts_to_sequences(texts)
    text_sequences = pad_sequences(text_sequences, maxlen=max_len, padding='post')
    
    embedding_matrix = load_glove_embeddings(word_index=tokenizer.word_index)

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)
    label_sequences = label_tokenizer.texts_to_sequences(labels)
    label_sequences = pad_sequences(label_sequences, maxlen=max_len, padding='post')
    label_sequences = to_categorical(label_sequences)
    
    word_index = tokenizer.word_index
    label_index = label_tokenizer.word_index
    
    return text_sequences, label_sequences, word_index, label_index, embedding_matrix