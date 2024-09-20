from logging import exception
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from flask_restful import Api, Resource
import matplotlib.pyplot as plt
import io
import lime
import lime.lime_text
import base64
import joblib
import preprocess
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)

model_svm = joblib.load("../Model/model_svm.joblib")

class_names = [0, 1, 2]
class_dict = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
sent_to_label = {'negative': 0, 'neutral': 1, 'positive': 2}
explainer = lime.lime_text.LimeTextExplainer(class_names=class_names)


def get_sentiment(text):
    return model_svm.predict([text])

def get_img_in_buffer(img):
    buf = io.BytesIO()
    img.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# --> Endpoint for Classification
class PredictSentiment(Resource):
    def post(self):
        data = request.json
        text = data['text']
        text = preprocess.preprocess_text(text)
        prediction = get_sentiment(text)
        sentiment = 'Positive' if prediction == 2 else 'Negative' if prediction == 0 else 'Neutral'
        return jsonify({'text': text, 'sentiment': sentiment})

# --> Endpoint for explanation
class ExplainPrediction(Resource):
    def post(self):
        data = request.json
        text = data['text']
        exp = explainer.explain_instance(
            text, model_svm.predict_proba, num_features=100, top_labels=5)
        text = preprocess.preprocess_text(text)
        prediction = get_sentiment(text)
        label = 2 if prediction == 2 else 0 if prediction == 0 else 1
        fig = exp.as_pyplot_figure(label=label)
        html_output = exp.as_html(text=True)
        ls = exp.as_list(label=label)
        list_html = "<ul>"
        for feature, weight in ls:
            list_html += f"<li>{feature}: {weight:.4f}</li>"
        list_html += "</ul>"
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_base64=base64.b64encode(buf.getvalue()).decode('utf-8')

        return jsonify({'text': text, 'explanation': img_base64, 'html': html_output, 'pred': class_dict[label], 'list_html': list_html})


class GetDashboard(Resource):
    def post(self):
        try:
            if 'dataset' not in request.files or 'text_feild' not in request.form:
                return jsonify({"error": "Missing file or text field name"}), 400

            dataset = request.files['dataset']
            text_field = request.form['text_feild']
            print('Got data')
            df = pd.read_csv(dataset, encoding='utf-8')
            print(df.head())
            print(text_field, type(text_field))
            if text_field not in df.columns:
                return jsonify({"error": f"Column '{text_field}' not found in the CSV file"}), 400

            X = df[text_field].apply(preprocess.preprocess_text)
            print('data_got_preproceesed')
            y_pred = model_svm.predict(X)
            print('got prediction')
            df['prediction'] = y_pred

            desired_location = '../temp/data.csv'
            df.to_csv(desired_location, index=False)

            count = df['prediction'].value_counts()
            neg = int(count.get(0, 0))
            neu = int(count.get(1, 0))
            pos = int(count.get(2, 0))
            print(neg, neu, pos, df.shape[0])

            negative_texts = df[df['prediction'] == 0][text_field].sample(5, replace=True).tolist()
            neutral_texts = df[df['prediction'] == 1][text_field].sample(5, replace=True).tolist()
            positive_texts = df[df['prediction'] == 2][text_field].sample(5, replace=True).tolist()
            # print(negative_texts)
            # print(neutral_texts)
            # print(positive_texts)

            # Create bar plot
            fig1, ax1 = plt.subplots()
            sns.barplot(x=['Negative', 'Neutral', 'Positive'], y=[neg, neu, pos], ax=ax1)
            ax1.set_title('Sentiment Analysis Bar Chart')
            ax1.set_xlabel('Sentiment')
            ax1.set_ylabel('Count')
            fig1_base64 = get_img_in_buffer(fig1)

            # Create pie chart
            fig2, ax2 = plt.subplots()
            ax2.pie([neg, neu, pos], labels=['Negative', 'Neutral', 'Positive'], autopct='%1.1f%%')
            ax2.set_title('Sentiment Analysis Pie Chart')
            fig2_base64 = get_img_in_buffer(fig2)
            plt.close(fig2)
            plt.close(fig1)
            print('done till here')

            return jsonify({
                "total_count": int(df.shape[0]),
                "neg_count": neg,
                "pos_count": pos,
                "neu_count": neu,
                "sample_text": {
                    "negative": negative_texts,
                    "neutral": neutral_texts,
                    "positive": positive_texts
                },
                "bar_plot": fig1_base64,
                "pie_chart": fig2_base64
            })
        except Exception as e:
            print(e)
            return jsonify({"error": str(e)}), 500


class GetSentimentData(Resource):
    def post(self):
        data = request.json
        sentiment = str(data['sentiment']).lower()
        label = sent_to_label.get(sentiment)

        if label is None:
            return jsonify({"error": "Invalid sentiment type"}), 400

        try:
            df = pd.read_csv('../temp/data.csv')
        except Exception as e:
            return jsonify({"error": f"We are facing some error: {e}"}), 500

        if sentiment == 'total':
            send_df = df
        else:
            send_df = df[df['prediction'] == label]

        csv_string = send_df.to_csv(index=False)

        return Response(csv_string, mimetype='text/csv', headers={
            'Content-Disposition': f'attachment; filename="{sentiment}_sentiment_text.csv"'
        })


api.add_resource(PredictSentiment, '/predict')
api.add_resource(ExplainPrediction, '/explain')
api.add_resource(GetDashboard, '/db')
api.add_resource(GetSentimentData, '/download')

if __name__ == '__main__':
    app.run(debug=True)
