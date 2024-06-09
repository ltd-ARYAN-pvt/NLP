from flask import Flask, request, jsonify
import tensorflow as tf
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import numpy as np

app = Flask(__name__)
CORS(app)
# # Load the trained model
# model = load_model('path/to/your/trained_model.h5')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    print(text)
    dialogue_acts=["sg","yn","+"]
    return jsonify({'dialogue_acts': dialogue_acts})

if __name__ == '__main__':
    app.run(debug=True)