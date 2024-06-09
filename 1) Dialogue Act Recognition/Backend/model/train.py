import config
import dataset
from model import create_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from convokit import Corpus, download
from sklearn.model_selection import train_test_split

#--> Getting Dataset
corpus = Corpus(filename=download("switchboard-processed-corpus"))
raw_data=corpus.get_utterances_dataframe().copy()
print(raw_data.head())
raw_data.drop(['timestamp', 'text', 'speaker', 'reply_to', 'conversation_id', 'meta.prev_id','meta.next_id', 'vectors','meta.full_tags'],axis=1, inplace=True)
raw_data.reset_index(drop=True, inplace=True)
raw_data.rename(columns={'meta.alpha_text': 'text', 'meta.tags': 'labels'}, inplace=True)
print(raw_data.sample(5))

#--> Converting Dataset to embedding
texts,labels=dataset.get_texts_labels(raw_data,'text','labels')
print(len(texts),len(labels))

text_sequences, label_sequences, word_index, label_index, embedding_matrix=dataset.preprocess_text_pipeline(texts,labels)
print(len(text_sequences),len(label_sequences),len(label_index),len(word_index),len(embedding_matrix))
print(text_sequences)
print(label_sequences)

#--> Model Training
num_words = len(word_index) + 1
num_labels = len(label_index) + 1
model=create_model(embedding_matrix,num_words,config.max_len,config.lstm_units1,config.lstm_units2, num_labels)


batch_size = config.batch_size
epochs = config.epochs

X_train, X_val, y_train, y_val = train_test_split(text_sequences, label_sequences, test_size=0.2, random_state=42)


history = model.fit([X_train, X_train], y_train,
                    validation_data=([X_val, X_val], y_val),
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=config.callbacks)