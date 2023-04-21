# pipreqs --force to update requirements.txt
import streamlit as st
import json
import requests
from streamlit_lottie import st_lottie
import constants
import utils
import ssl




def load_lottieUrl(url):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()


def setPageConfig():
    st.set_page_config(
        page_title="Spam AI Checker Demo", page_icon=":brain:", layout="wide"
    )
    st.markdown("## Introduction 💡")
    st.sidebar.markdown("## Introduction 💡")
    utils.common_styling()
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    co = utils.getCohereApiClient()


def setPageHeaders():
    with st.container():
        st.subheader("Hi, I am Simon from MB Office :wave:")
        st.title("SMS Spam AI Assistant")
        st.markdown("Quickly identify any SMS text message as spam")
        st.markdown(
            """[Learn More out the dataset >](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)"""
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
            data = load_lottieUrl(constants.LOTTIE_AI_JSON)
            st_lottie(data, height=300, key="ai")

setPageConfig()
setPageHeaders()
setPageBodyLayout()