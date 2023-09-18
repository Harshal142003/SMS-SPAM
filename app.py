import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from flask import Flask, render_template, request, jsonify


# Load NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Porter Stemmer
ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load TF-IDF Vectorizer
with open('vectorizer2.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

with open('model2.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        input_sms = request.form['message']

        if not input_sms:
            return jsonify({'result': 'Please enter a message.'})

        # Preprocess
        transformed_sms = transform_text(input_sms)
        # Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # Predict
        result = model.predict(vector_input)

        if result == 1:
            result_text = "Spam"
        else:
            result_text = "Not Spam"

        return jsonify({'result': result_text})

    return render_template('index.html', result="")

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=5000)
