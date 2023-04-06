# pipreqs --force to update requirements.txt


import numpy as np
import pandas as pd
import cohere as cohere
from cohere.responses.classify import Example
import streamlit as st
import json
from array import array
import requests
from streamlit_lottie import st_lottie
import random
import charset_normalizer
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

# VARIABLES
DATA_SOURCE = "dataset/spam.csv"
MAX_ROWS = 1000
LOTTIE_AI_JSON = "https://assets2.lottiefiles.com/packages/lf20_zrqthn6o.json"
API_KEY = st.secrets["API_KEY"]
co = cohere.Client(API_KEY)
textMessage = ""
AI_PROMPT_SPAM = "write me a sample SMS text message that is a spam message and less than 150 characters"
AI_PROMPT_NON_SPAM = "write me a sample SMS text message that is not a spam message and less than 150 characters"
DATA_CATEGORY_TYPE_SPAM = "spam"
DATA_CATEGORY_TYPE_NON_SPAM = "ham"
FORMATTED_TXT = """<span style="word-wrap:break-word;">{content}</span>"""

# Initialization
if "message" not in st.session_state:
    st.session_state["message"] = ""

if "classify_inputs" not in st.session_state:
    st.session_state["classify_inputs"] = []

if "json_objects" not in st.session_state:
    st.session_state["json_objects"] = []


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_SOURCE)
    if (data.shape[0]) > MAX_ROWS:
        df = data.head(MAX_ROWS)
    else:
        df = data
    return df


@st.cache_data
def load_all_data():
    with st.container():
        st.write("---")
        # attempting to detect the csv file character encoding
        with open(DATA_SOURCE, "rb") as csv:
            detected_encoding = charset_normalizer.detect(csv.read(250000))
        df = pd.read_csv(DATA_SOURCE, encoding=detected_encoding["encoding"])
        left_col, right_col = st.columns(2)
        with right_col:
            spam_text_list = df.query("Category == 'spam'")["Message"].values.tolist()
            spam_text_collection = " ".join(spam_text_list)
            ham_text_list = df.query("Category == 'ham'")["Message"].values.tolist()
            ham_text_collection = " ".join(ham_text_list)
            # Create and generate a word cloud image:
            wordcloud_spam = WordCloud(
                width=520,
                height=260,
                stopwords=STOPWORDS,
                max_font_size=50,
                background_color="black",
                colormap="Blues",
            ).generate(spam_text_collection)
            wordcloud_ham = WordCloud(
                width=520,
                height=260,
                stopwords=STOPWORDS,
                max_font_size=50,
                background_color="black",
                colormap="Greens",
            ).generate(ham_text_collection)
            # Display the generated image:
            plt.title("Spam Word Cloud")
            plt.imshow(wordcloud_spam, interpolation="bilinear")
            plt.axis("off")
            plt.show()
            st.pyplot()
            plt.title("Non Spam Word Cloud")
            plt.imshow(wordcloud_ham, interpolation="bilinear")
            plt.axis("off")
            plt.show()
            st.pyplot()
        with left_col:
            st.header("Data Visualization :bar_chart:")
            st.write("##")
            counts_by_Category = (
                df.groupby(by=["Category"])
                .count()[["Message"]]
                .sort_values(by="Category")
            )
            fig_spam_check = px.bar(
                counts_by_Category,
                x=counts_by_Category.index,
                y="Message",
                orientation="v",
                title="<b>Text Message by Category</b>",
                color_discrete_sequence=["#0083B8"] * len(counts_by_Category),
                template="plotly_white",
                labels={
                    "Message": "Message Count",
                    "Category": "Message Category",
                },
            )
            config = {
                "toImageButtonOptions": {
                    "format": "svg",  # one of png, svg, jpeg, webp
                    "filename": "custom_image",
                    "height": 500,
                    "width": 700,
                    "scale": 1,  # Multiply title/legend/axis/canvas sizes by this factor
                },
                "displaylogo": False,
                "modeBarButtonsToRemove": ["zoom", "pan"],
            }
            fig_spam_check.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(showgrid=False),
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.set_option("deprecation.showPyplotGlobalUse", False)
            st.plotly_chart(fig_spam_check, use_container_width=False, config=config)


def load_lottieUrl(url):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()


def setPageConfig():
    st.set_page_config(
        page_title="Spam AI Checker Demo", page_icon=":brain:", layout="wide"
    )


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
setDataVisualization()
setClassificationLogic()
