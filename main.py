# pipreqs --force to update requirements.txt


import numpy as np
import pandas as pd
import cohere as cohere
from cohere.responses.classify import Example
import streamlit as st
import json
import requests
from streamlit_lottie import st_lottie
import random
import plotly.express as px
from dotenv import dotenv_values
import dask.dataframe as dd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

import re
import nltk
import constants
# VARIABLES
DATA_SOURCE = "dataset/spam.csv"
MAX_ROWS = 1000
LOTTIE_AI_JSON = "https://assets2.lottiefiles.com/packages/lf20_zrqthn6o.json"
config = dotenv_values(".env")
API_KEY = config["API_KEY"]
co = cohere.Client(API_KEY)
textMessage = ""
AI_PROMPT_SPAM = "write me a sample SMS text message that is a spam message and less than 150 characters"
AI_PROMPT_NON_SPAM = "write me a sample SMS text message that is not a spam message and less than 150 characters"
DATA_CATEGORY_TYPE_SPAM = "spam"
DATA_CATEGORY_TYPE_NON_SPAM = "ham"
FORMATTED_TXT = """<span style="word-wrap:break-word;">{content}</span>"""
primary_blue = "#496595"
primary_blue2 = "#85a1c1"
primary_blue3 = "#3f4d63"
primary_grey = "#c6ccd8"
# Initialization
if "message" not in st.session_state:
    st.session_state["message"] = ""

if "classify_inputs" not in st.session_state:
    st.session_state["classify_inputs"] = []

if "json_objects" not in st.session_state:
    st.session_state["json_objects"] = []

def clean(Message):
    # Replacing all non-alphabetic characters with a space
    sms = re.sub('[^a-zA-Z]', ' ', Message) 
    #converting to lowecase
    sms = sms.lower() 
    sms = sms.split()
    sms = ' '.join(sms)
    return sms

@st.cache_data
def load_data(rowNumber=MAX_ROWS):
    data = dd.read_csv(DATA_SOURCE)
    data = data.compute()
    data["Message"] = data["Message"].apply(clean)
    if (data.shape[0]) > rowNumber:
        df = data.sample(rowNumber)
    else:
        df = data
    return df

def generate_word_cloud(colorMap, textCollection):
    cloud = WordCloud(
                width=520,
                height=260,
                stopwords=STOPWORDS,
                max_font_size=50,
                background_color="black",
                colormap=colorMap,
            ).generate(textCollection)





def load_lottieUrl(url):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()


def setPageConfig():
    st.set_page_config(
        page_title="Spam AI Checker Demo", page_icon=":brain:", layout="wide"
    )
    st.markdown("# Intro 💡")
    st.sidebar.markdown("# Intro 💡")


def setPageHeaders():
    with st.container():
        st.subheader("Hi, I am Simon from MB Office :wave:")
        st.title("SMS Spam AI Assistant")
        st.markdown("Quickly identify any SMS text message as spam")
        st.markdown(
            """[Learn More out training dataset >](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)"""
        )


def setPageBodyLayout():
    with st.container():
        st.write("---")
        left_col, right_col = st.columns(2)
        with left_col:
            st.header("Motivation :thinking_face:")
            st.write("##")
            st.write(
                """
            Scammers send fake text messages to trick you into giving them your personal information — things like your password, account number, or Social Security number. If they get that information, they could gain access to your email, bank, or other accounts. Or they could sell your information to other scammers.
            """
            )
        with right_col:
            data = load_lottieUrl(LOTTIE_AI_JSON)
            st_lottie(data, height=300, key="ai")


def loadExamples(data):
    examples = []
    df2 = data.to_json(orient="records")
    json_object = json.loads(df2)
    st.session_state["json_objects"] = json_object
    for item in json_object:
        examples.append(Example(item["Message"], item["Category"]))
    return examples


def generateRandomMessageByAI(prompt):
    data_load_state = st.text("start text generation...")
    response = co.generate(
        model="command-xlarge-nightly",
        prompt=prompt,
        max_tokens=100,
        temperature=0.9,
        k=0,
        stop_sequences=[],
        return_likelihoods="NONE",
    )
    if len(response.generations) > 0:
        text_message = str(response.generations[0].text)
        classify_inputs = []
        classify_inputs.append(text_message)
        st.session_state["classify_inputs"] = classify_inputs
        st.write(text_message)
    data_load_state.text("")


def generateRandomMessageComponent(type):
    with st.container():
        columns1, columns2 = st.columns(2)
        spam_message_button = columns1.button(
            "Generate Spam Message", key=DATA_CATEGORY_TYPE_SPAM, type="primary"
        )
        ham_message_button = columns2.button(
            "Generate Non-Spam Message",
            key=DATA_CATEGORY_TYPE_NON_SPAM,
            type="secondary",
        )
        if spam_message_button:
            if type == "AI":
                generateRandomMessageByAI(AI_PROMPT_SPAM)
            else:
                generateRandomMessageByDataSet(DATA_CATEGORY_TYPE_SPAM)
        if ham_message_button:
            if type == "AI":
                generateRandomMessageByAI(AI_PROMPT_NON_SPAM)
            else:
                generateRandomMessageByDataSet(DATA_CATEGORY_TYPE_NON_SPAM)


def filterStatusBySpam(data):
    if data["Category"] == DATA_CATEGORY_TYPE_SPAM:
        return True
    return False


def filterStatusByNonSpam(data):
    if data["Category"] == DATA_CATEGORY_TYPE_NON_SPAM:
        return True
    return False


def generateRandomMessageByDataSet(type):
    if type == DATA_CATEGORY_TYPE_SPAM:
        dataset = st.session_state["json_objects"]
        spamList = list(filter(filterStatusBySpam, dataset))
        text_message = random.choice(spamList)["Message"]
        classify_inputs = []
        classify_inputs.append(text_message)
        st.write("##")
        st.markdown(text_message)
        st.session_state["classify_inputs"] = classify_inputs
    if type == DATA_CATEGORY_TYPE_NON_SPAM:
        dataset = st.session_state["json_objects"]
        nonSpamList = list(filter(filterStatusByNonSpam, dataset))
        text_message = random.choice(nonSpamList)["Message"]
        classify_inputs = []
        classify_inputs.append(text_message)
        st.write("##")
        st.write(text_message)
        st.session_state["classify_inputs"] = classify_inputs


def setDataVisualization():
    load_all_data()

def print_estimator_name(estimator):
    return estimator.__class__.__name__

@st.cache_data
def setEmbeddedClassification():
    # Splitting the testing and training sets
    # Build a pipeline of model for four different classifiers.
    # RandomForestClassifier
    # KNeighborsClassifier
    # Support Vector Machines
    # Fit all the models on training data
    # Get the cross-validation on the training set for all the models for accuracy
    with st.container():
         st.write("---")
         st.header("Classification Comparison")
         left_col, right_col = st.columns(2)
    #Testing on the following classifiers
         with left_col:
            df_sample = load_data()
            sms_train, sms_test, labels_train, labels_test = train_test_split(
            list(df_sample["Message"]), list(df_sample["Category"]), test_size=0.25, random_state=42)
            embeddings_train_large = co.embed(texts=sms_train,
                             model="large",
                             truncate="RIGHT").embeddings
            embeddings_test_large = co.embed(texts=sms_test,
                             model="large",
                             truncate="RIGHT").embeddings
            embeddings_train_small = co.embed(texts=sms_train,
                             model="small",
                             truncate="RIGHT").embeddings
            embeddings_test_small = co.embed(texts=sms_test,
                             model="small",
                             truncate="RIGHT").embeddings                            
            classifiers = [ RandomForestClassifier(),
                            KNeighborsClassifier(), 
                            SVC()]
            for classifier in classifiers:
                classifier.fit(embeddings_train_large, labels_train)     
                score = classifier.score(embeddings_test, labels_test)
                st.write(f"{print_estimator_name(classifier)} Validation accuracy on Large is {100*score}%!")

def setClassificationLogic():
    data = load_data()
    examples = loadExamples(data)
    with st.container():
        st.write("---")
        st.header("Classification :rocket:")
        left_col, right_col = st.columns(2)
        with left_col:
            option = st.selectbox(
                "How would you like generate a text message",
                ("", "From existing dataset", "AI Generated", "Manual Input"),
            )
            if option == "From existing dataset":
                generateRandomMessageComponent("Dataset")
            elif option == "Manual Input":
                with st.container():
                    input = st.text_area(
                        label="Text Message",
                        value="",
                        placeholder="Input your sms text message here",
                        label_visibility="hidden",
                    )
                    text_message = str(input)
                    classify_inputs = []
                    classify_inputs.append(text_message)
                    st.session_state["classify_inputs"] = classify_inputs
            elif option == "AI Generated":
                generateRandomMessageComponent("AI")

            with st.container():
                classify = st.button(
                    "Start classify",
                    type="primary",
                    disabled=False
                    if len(st.session_state["classify_inputs"]) > 0
                    else True,
                )
                if classify:
                    data_load_classification = st.text("Start Classification...")
                    response = co.classify(
                        model="large",
                        inputs=st.session_state["classify_inputs"],
                        examples=examples,
                    )
                    st.write(st.session_state["classify_inputs"][0])
                    if len(response.classifications) > 0:
                        result = (
                            "Above message is not a spam message"
                            if response.classifications[0].prediction
                            == DATA_CATEGORY_TYPE_NON_SPAM
                            else "Above message is a spam message"
                        )
                        st.write(result)
                        st.write(
                            "Confidence value is: {accuracy}".format(
                                accuracy=response.classifications[0].confidence
                            )
                        )

                    data_load_classification.text("")


setPageConfig()
setPageHeaders()
setPageBodyLayout()
# setDataVisualization()
# setEmbeddedClassification()
# setClassificationLogic()
