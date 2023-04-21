import streamlit as st
from dotenv import dotenv_values
import cohere as cohere
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import re

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
nltk.download('wordnet')

def common_styling():       
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def getCohereApiClient():
    config = dotenv_values(".env")
    API_KEY = config["API_KEY"]
    co = cohere.Client(API_KEY)
    return co


def clean(Message):
    # Replacing all non-alphabetic characters with a space
    sms = re.sub('[^a-zA-Z]', ' ', Message) 
    #converting to lowecase
    sms = sms.lower() 
    sms = sms.split()
    sms = ' '.join(sms)
    return sms

def print_estimator_name(estimator):
    return estimator.__class__.__name__

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    filtered_text = [word for word in text if word not in stop_words]
    return filtered_text

def tokenize_word(text):
    tokens = [nltk.word_tokenize(text)]
    return tokens

def lemmatize_word(text):
    #word_tokens = word_tokenize(text)
    # provide context i.e. part-of-speech
    lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in text]
    return lemmas
