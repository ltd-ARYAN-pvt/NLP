from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import numpy as np

lr=3e-3
glove_path='../jupyter Notebook/glove.6B.50d.txt'
max_len = 50
lstm_units1 = 128
lstm_units2 = 128
batch_size = 64
epochs = 100
embedding_dim=50

checkpoint = ModelCheckpoint('best_model_s2s.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.000001, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)

callbacks = [checkpoint, reduce_lr, early_stopping]

desired_dialogue_acts = [
    'sd', 'sv', '%', 'b', 'aa', '+', 'qy', 'ba','qw', 'ny', 'h', 'fc', 'nn', 'bf','na','qo','o'
]

desired_dialogue_dict={
    'sd':"Statement-non-opinion",
    'sv':"Statement-opinion",
    '%':"Abandoned or Turn-Exit",
    'b':"Acknowledge (Backchannel)",
    'aa':"Accept",
    '+':"Segment (multi-utterance)",
    'qy':"Yes-No-Question",
    'ba':"Appreciation",
    'qw':"Wh-Question",
    'ny':"Yes answers",
    'h':"Hedge",
    'fc':"Conventional-closing",
    'nn':"No answers",
    'bf':"Summarize/reformulate",
    'na':"Affirmative non-yes answers",
    'qo':"Open-Question",
    'o':"other",
}

def filter_labels(texts, labels, desired_labels=desired_dialogue_acts):
    filtered_texts = []
    filtered_labels = []
    for text, label in zip(texts, labels):
        #--> Split labels if there are multiple labels for a single text
        label_list = label.strip().split()
        filtered_label_list = [lbl for lbl in label_list if lbl in desired_labels]
        if filtered_label_list:
            filtered_texts.append(text)
            filtered_labels.append(" ".join(filtered_label_list))
    return filtered_texts, filtered_labels

def inverse_transform_predictions(predictions, label_index):
    #--> Create an inverse mapping from indices to labels
    inverse_label_index = {v: k for k, v in label_index.items()}
    
    #--> Iterate over the predictions and convert them back to labels
    transformed_predictions = []
    for sequence in predictions:
        transformed_sequence = [inverse_label_index[np.argmax(token)] for token in sequence if np.argmax(token) in inverse_label_index]
        transformed_predictions.append(transformed_sequence)
    
    return transformed_predictions