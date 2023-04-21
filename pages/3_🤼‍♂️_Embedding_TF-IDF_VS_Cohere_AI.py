import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import dask.dataframe as dd
import matplotlib.pyplot as plt
import utils
import constants
import nltk
import pandas as pd
from datetime import datetime
# import warnings filter
from warnings import simplefilter
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
nltk.download('omw-1.4')
nltk.download('punkt')
st.markdown("## 🤼‍♂️ Embedding Comparison")
st.sidebar.markdown("## 🤼‍♂️ Embedding Comparison")
utils.common_styling()

accuracy = {}


@st.cache_data(persist="disk", show_spinner=True)
def load_data(rowNumber=2000):
    data = dd.read_csv(constants.DATA_SOURCE)
    data = data.compute()
    # clean data
    data["Message"] = data["Message"].apply(utils.clean)
    # sampling the data
    if (data.shape[0]) > rowNumber:
        data = data.groupby('Category', group_keys=False).apply(lambda x: x.sample(frac=0.2))
    else:
        data = data
    return data

@st.cache_data(persist="disk", show_spinner=True)
def load_data_with_nltk_preprocessing(rowNumber=constants.MAX_ROWS):
    data = dd.read_csv(constants.DATA_SOURCE)
    data = data.compute()
    # clean data

    data["Message"] = data["Message"].apply(utils.clean)
    # tokenization
    # Tokenization is breaking complex data into smaller units called tokens. It can be done by splitting paragraphs into sentences and sentences into words. I am splitting the Clean_Text into words at this step.
    data["Tokenize_Text"]=data.apply(lambda row: nltk.word_tokenize(row["Message"]), axis=1)
    # data["Message"] = data["Message"].apply(utils.tokenize_word)
    # Stop words are frenquently workds that do not contribute much to NLP.
    data["Tokenize_Text"] =  data["Tokenize_Text"].apply(utils.remove_stopwords)
    # lemmatization converts a word to its root form and ensure the root word belongs to the language
    data["Message"] = data["Tokenize_Text"].apply(utils.lemmatize_word)
    # sampling the data
    if (data.shape[0]) > rowNumber:
        data = data.sample(rowNumber)
    else:
        data = data
    return data

@st.cache_data(persist="disk", show_spinner=True)
def setEmbeddedClassificationTFIDF():
    t1 = datetime.now()
    data = load_data_with_nltk_preprocessing()
    corpus = []
    for i in data["Message"]:
        msg = ' '.join([row for row in i])
        corpus.append(msg)
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(corpus).toarray()
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(data["Category"])
    # Splitting the testing and training sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    classifiers = [
               RandomForestClassifier(),
               KNeighborsClassifier(), 
               SVC()]
    tfidf_accuracy = {}
    for classifier in classifiers:
        classifier.fit(X_train, Y_train)     
        score = classifier.score(X_test, Y_test)
        tfidf_accuracy[utils.print_estimator_name(classifier)] = score
    t2 = datetime.now()
    delta = t2 - t1
    st.write(f"Classification using TF-IDF takes {delta.total_seconds()} seconds")
    return tfidf_accuracy

@st.cache_data(persist="disk", show_spinner=True)
def setEmbeddedClassificationCohere():
    # Splitting the testing and training sets
    # Build a pipeline of model for four different classifiers.
    # RandomForestClassifier
    # KNeighborsClassifier
    # Support Vector Machines
    # Fit all the models on training data
    # Get the cross-validation on the training set for all the models for accuracy
    with st.container():
         left_col, right_col = st.columns(2)
    #Testing on the following classifiers
         with left_col:
            t1 = datetime.now()
            df_sample = load_data()
            sms_train, sms_test, labels_train, labels_test = train_test_split(
            list(df_sample["Message"]), list(df_sample["Category"]), test_size=0.2, random_state=42, stratify=list(df_sample["Category"]))
            st.write(sms_train)
            embeddings_train_small = co.embed(texts=sms_train,
                             model="small",
                             truncate="RIGHT").embeddings
            embeddings_test_small = co.embed(texts=sms_test,
                             model="small",
                             truncate="RIGHT").embeddings                            
            classifiers = [ RandomForestClassifier(),
                            KNeighborsClassifier(), 
                            SVC()]
            cohere_small_list = {}
            for classifier in classifiers:
                classifier.fit(embeddings_train_small, labels_train)     
                score = classifier.score(embeddings_test_small, labels_test)
                cohere_small_list[utils.print_estimator_name(classifier)] = score
            t2 = datetime.now()
            delta = t2 - t1
            st.write(f"Classification using Cohere takes {delta.total_seconds()} seconds")
            return {
                'Cohere Small Model': cohere_small_list 
                }
           
my_dict = {}
my_dict["TF-IDF"] = setEmbeddedClassificationTFIDF()
cohereData = setEmbeddedClassificationCohere()

my_dict["Cohere Small Model"] = cohereData["Cohere Small Model"]
df = pd.DataFrame(my_dict)
df