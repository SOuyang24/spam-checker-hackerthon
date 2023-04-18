import streamlit as st
import constants
import random
import plotly.express as px
from sklearn import metrics
from cohere.responses.classify import Example
import dask.dataframe as dd
import matplotlib.pyplot as plt
import utils
import json
from datetime import datetime

co = utils.getCohereApiClient()
utils.common_styling()
if "message" not in st.session_state:
    st.session_state["message"] = ""

if "classify_inputs" not in st.session_state:
    st.session_state["classify_inputs"] = []

if "json_objects" not in st.session_state:
    st.session_state["json_objects"] = []

st.markdown("### 🤖 Spam Prediction")
st.sidebar.markdown("### 🤖 Spam Prediction")

@st.cache_data
def load_data(rowNumber=constants.MAX_ROWS):
    data = dd.read_csv(constants.DATA_SOURCE)
    data = data.compute()
    if (data.shape[0]) > rowNumber:
        df = data.sample(rowNumber)
    else:
        df = data
    return df

def filterStatusBySpam(data):
    if data["Category"] == constants.DATA_CATEGORY_TYPE_SPAM:
        return True
    return False


def filterStatusByNonSpam(data):
    if data["Category"] == constants.DATA_CATEGORY_TYPE_NON_SPAM:
        return True
    return False

def generateRandomMessageByAI(prompt):
    data_load_state = st.text("start text generation...")
    classify_inputs = []
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
        classify_inputs.append(text_message)
        st.session_state["classify_inputs"] = classify_inputs
        st.write(text_message)
    data_load_state.text("")

def generateRandomMessageByDataSet(type):
    if type == constants.DATA_CATEGORY_TYPE_SPAM:
        dataset = st.session_state["json_objects"]
        spamList = list(filter(filterStatusBySpam, dataset))
        text_message = random.choice(spamList)["Message"]
        classify_inputs = []
        classify_inputs.append(text_message)
        st.write("##")
        st.markdown(text_message)
        st.session_state["classify_inputs"] = classify_inputs
    if type == constants.DATA_CATEGORY_TYPE_NON_SPAM:
        dataset = st.session_state["json_objects"]
        nonSpamList = list(filter(filterStatusByNonSpam, dataset))
        text_message = random.choice(nonSpamList)["Message"]
        classify_inputs = []
        classify_inputs.append(text_message)
        st.write("##")
        st.write(text_message)
        st.session_state["classify_inputs"] = classify_inputs

def generateRandomMessageComponent(type):
    with st.container():
        inputs = []
        columns1, columns2 = st.columns(2)
        spam_message_button = columns1.button(
            "Generate Spam SMS", key=constants.DATA_CATEGORY_TYPE_SPAM, type="primary"
        )
        ham_message_button = columns2.button(
            "Generate Normal SMS",
            key=constants.DATA_CATEGORY_TYPE_NON_SPAM,
            type="secondary",
        )
        if spam_message_button:
            if type == "AI":
                generateRandomMessageByAI(constants.AI_PROMPT_SPAM)
            else:
                generateRandomMessageByDataSet(constants.DATA_CATEGORY_TYPE_SPAM)
        if ham_message_button:
            if type == "AI":
                generateRandomMessageByAI(constants.AI_PROMPT_NON_SPAM)
            else:
                generateRandomMessageByDataSet(constants.DATA_CATEGORY_TYPE_NON_SPAM)

def load_examples(data):
    examples = []
    df2 = data.to_json(orient="records")
    json_object = json.loads(df2)
    st.session_state["json_objects"] = json_object
    for item in json_object:
        examples.append(Example(item["Message"], item["Category"]))
    st.session_state["examples"] = examples

def setClassificationLogic():
    inputs = []
    classify_inputs = []
    data = load_data()
    load_examples(data)
    with st.container():
        left_col, right_col = st.columns([3, 1])
        with left_col:
            option = st.selectbox(
                "How would you like generate a text message?",
                ("", "From existing dataset", "AI Generated", "Manual Input"),
            )
            if option == "From existing dataset":
                # st.session_state["classify_inputs"] = []
                generateRandomMessageComponent("Dataset")
            elif option == "Manual Input":
                with st.container():
                    text_message = ""
                    input = st.text_area(
                        label="Text Message",
                        value="",
                        placeholder="Input your sms text message here",
                        label_visibility="hidden",
                    )
                    text_message = str(input)
                    inputs.append(text_message)
                    st.session_state["classify_inputs"] = inputs
            elif option == "AI Generated":
                # st.session_state["classify_inputs"] = []
                generateRandomMessageComponent("AI")

            with st.container():
                classify = st.button(
                    "Start classify",
                    type="primary",
                    disabled=False
                    if len(st.session_state["classify_inputs"]) > 0
                    else True,
                    # on_click=handle_classification
                )
                if classify:
                    t1 = datetime.now()
                    data_load_classification = st.text("Start Classification...")
                    response = co.classify(
                        model="small",
                        inputs=st.session_state["classify_inputs"],
                        examples=st.session_state["examples"],
                    )
                    st.write(st.session_state["classify_inputs"][0])
                    if len(response.classifications) > 0:
                        result = (
                            "Above message is not a spam message"
                            if response.classifications[0].prediction
                            == constants.DATA_CATEGORY_TYPE_NON_SPAM
                            else "Above message is a spam message"
                        )
                        st.write(result)
                        st.write(
                            "Confidence value is: {accuracy}".format(
                                accuracy=response.classifications[0].confidence
                            )
                        )
                    t2 = datetime.now()
                    delta = t2 - t1
                    st.write(f"Total prediction takes {delta.total_seconds()} seconds")
                    data_load_classification.text("")

setClassificationLogic()