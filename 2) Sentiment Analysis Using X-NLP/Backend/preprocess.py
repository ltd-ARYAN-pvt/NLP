import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

negative_words = {"don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "cannot", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't", "mustn't", "needn't", "shan't", "shouldn't",'not', "mightn't", "couldn't",'no', "hasn't"}

stop_words -= negative_words

def preprocess_text(text):
    # 1. Lowercasing
    text = str(text).lower()

    # 2. Remove HTML tags and special characters
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 3. Tokenize the text
    tokens = word_tokenize(text)
    # print(tokens)

    # 4. Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    # print(tokens)

    # 5. Lemmatize the tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # print(tokens)

    # 7. Join the tokens back into a string
    text = ' '.join(tokens)

    return text
