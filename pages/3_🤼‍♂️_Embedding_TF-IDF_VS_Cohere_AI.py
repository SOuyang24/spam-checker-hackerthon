import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, plot_confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn import metrics
import dask.dataframe as dd
import matplotlib.pyplot as plt
import utils
import constants

st.markdown("## 🤼‍♂️ Embedding Comparison")
st.sidebar.markdown("## 🤼‍♂️ Embedding Comparison")
utils.common_styling()

co = utils.getCohereApiClient()

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
         st.header("NTLK")
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
                score = classifier.score(embeddings_test_large, labels_test)
                st.write(f"{utils.print_estimator_name(classifier)} Validation accuracy on Large is {100*score}%!")


setEmbeddedClassification()
