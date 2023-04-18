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


co = utils.getCohereApiClient()

if "message" not in st.session_state:
    st.session_state["message"] = ""

if "classify_inputs" not in st.session_state:
    st.session_state["classify_inputs"] = []

if "json_objects" not in st.session_state:
    st.session_state["json_objects"] = []

@st.cache_data
def load_data(rowNumber=constants.MAX_ROWS):
    data = dd.read_csv(constants.DATA_SOURCE)
    data = data.compute()
    data["Message"] = data["Message"].apply(utils.clean)
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
        columns1, columns2 = st.columns(2)
        spam_message_button = columns1.button(
            "Generate Spam Message", key=constants.DATA_CATEGORY_TYPE_SPAM, type="primary"
        )
        ham_message_button = columns2.button(
            "Generate Non-Spam Message",
            key=DATA_CATEGORY_TYPE_NON_SPAM,
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

def loadExamples(data):
    examples = []
    df2 = data.to_json(orient="records")
    json_object = json.loads(df2)
    st.session_state["json_objects"] = json_object
    for item in json_object:
        examples.append(Example(item["Message"], item["Category"]))
    return examples

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
                            == constants.DATA_CATEGORY_TYPE_NON_SPAM
                            else "Above message is a spam message"
                        )
                        st.write(result)
                        st.write(
                            "Confidence value is: {accuracy}".format(
                                accuracy=response.classifications[0].confidence
                            )
                        )

                    data_load_classification.text("")

st.markdown("# Page 4 ❄️")
st.sidebar.markdown("Spam Classification")
setClassificationLogic()