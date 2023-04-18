import streamlit as st
from dotenv import dotenv_values
import cohere as cohere
import re

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

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