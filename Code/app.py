import streamlit as st
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import nltk


ps = PorterStemmer()

tfidf = pickle.load(open("Vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for word in text:
        if word.isalnum():
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        if word not in stopwords.words("english") and word not in string.punctuation:
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        y.append(ps.stem(word))

    return " ".join(y)

st.title("Email/SMS Spam Classifier")

input_msg = st.text_area("Enter the message")

if st.button("Predict"):

    #preprocess

    transform_sms = transform_text(input_msg)

    #vecotorize

    vector_input = tfidf.transform([transform_sms])

    #modeling

    result = model.predict(vector_input)[0]

    #display

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")