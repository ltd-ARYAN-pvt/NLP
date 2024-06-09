from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_restful import Api, Resource
import matplotlib.pyplot as plt
import io
import lime
import lime.lime_text
import base64

app = Flask(__name__)
CORS(app)
api = Api(app)

class DummyModel:
    def predict(self, texts):
        return [1 if "good" in text else 0 for text in texts]

model = DummyModel()

class_names = ['negative', 'positive']
explainer = lime.lime_text.LimeTextExplainer(class_names=class_names)

class PredictSentiment(Resource):
    def post(self):
        data = request.json
        text = data['text']
        prediction = model.predict([text])[0]
        sentiment = 'positive' if prediction == 1 else 'negative'
        return jsonify({'text': text, 'sentiment': sentiment})

#--> Endpoint for explanation
class ExplainPrediction(Resource):
    def post(self):
        data = request.json
        text = data['text']
        exp = explainer.explain_instance(text, model.predict, num_features=6)
        fig = exp.as_pyplot_figure()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return jsonify({'text': text, 'explanation': img_base64})

api.add_resource(PredictSentiment, '/predict')
api.add_resource(ExplainPrediction, '/explain')

if __name__ == '__main__':
    app.run(debug=True)