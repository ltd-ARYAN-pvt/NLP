from numpy import average
import tensorflow as tf
from keras.layers import Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import F1Score
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Concatenate, Attention
import config as config

def create_model(embedding_matrix, num_words, max_len, lstm_units1, lstm_units2, num_labels):
    #--> Encoder
    encoder_inputs = Input(shape=(max_len,), name='encoder_inputs')
    encoder_embedding = Embedding(input_dim=num_words, output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], input_length=max_len, trainable=False, name='encoder_embedding')(encoder_inputs)
    
    encoder_lstm1 = LSTM(lstm_units1, return_sequences=True, return_state=True, name='encoder_lstm1')
    encoder_outputs1, state_h1, state_c1 = encoder_lstm1(encoder_embedding)
    
    encoder_lstm2 = LSTM(lstm_units2, return_state=True, name='encoder_lstm2')
    encoder_outputs2, state_h2, state_c2 = encoder_lstm2(encoder_outputs1)
    
    encoder_states = [state_h2, state_c2]

    #--> Decoder
    decoder_inputs = Input(shape=(max_len,), name='decoder_inputs')
    decoder_embedding = Embedding(input_dim=num_labels, output_dim=embedding_matrix.shape[1], mask_zero=True, name='decoder_embedding')(decoder_inputs)
    
    decoder_lstm1 = LSTM(lstm_units1, return_sequences=True, return_state=True, name='decoder_lstm1')
    decoder_outputs1, _, _ = decoder_lstm1(decoder_embedding, initial_state=[state_h1, state_c1])
    
    decoder_lstm2 = LSTM(lstm_units2, return_sequences=True, return_state=True, name='decoder_lstm2')
    decoder_outputs2, _, _ = decoder_lstm2(decoder_outputs1, initial_state=encoder_states)

    #--> Attention mechanism
    attention = Attention(name='attention_layer')
    # decoder_outputs2 = Reshape((-1, max_len, config.embedding_dim))(decoder_outputs2)
    # encoder_outputs2 = Reshape((-1, max_len, config.embedding_dim))(encoder_outputs2)

    #--> Error point
    context_vector = attention([decoder_outputs2, encoder_outputs2])
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs2, context_vector])

    #--> Output layer
    decoder_dense = TimeDistributed(Dense(num_labels, activation='softmax', name='output_layer'))
    decoder_outputs = decoder_dense(decoder_concat_input)

    #--> Define the model
    f1_score=F1Score()
    f1_score_macro=F1Score(average="macro")
    optimizer=Adam(learning_rate=config.lr)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', f1_score, f1_score_macro])
    model.summary()
    return model