import streamlit as st
import constants
import constants
import dask.dataframe as dd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
import utils
import nltk

st.markdown("## :bar_chart: Data Analysis")
st.sidebar.markdown("## :bar_chart: Data Analysis")
utils.common_styling()
 nltk.download('punkt')
@st.cache_data
def data_analysis():
    with st.container():
        df = dd.read_csv(constants.DATA_SOURCE)
        df = df.compute()
        left_col, right_col = st.columns(2)
        with left_col:
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
                # color_discrete_sequence=["#0083B8"] * len(counts_by_Category),
                color_discrete_map={"spam": "red", "ham": "green"},
                color=["ham", "spam"],
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

        left_col, right_col = st.columns(2)
        spam_text_list = df.query(f'Category == "{constants.DATA_CATEGORY_TYPE_SPAM}"')["Message"].values.tolist()
        spam_text_collection = " ".join(spam_text_list)
        ham_text_list = df.query(f'Category == "{constants.DATA_CATEGORY_TYPE_NON_SPAM}"')["Message"].values.tolist()
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
        st.markdown("### :cloud: Word Cloud")
        left_col2, right_col2 = st.columns(2)
        
        with left_col2: 
        # Display the generated image:
            fig1, ax = plt.subplots()
            plt.title("Spam Word Cloud")
            plt.imshow(wordcloud_spam, interpolation="bilinear")
            plt.axis("off")
            plt.show()
            st.pyplot(fig1)
        with right_col2:             
            fig2, ax = plt.subplots()
            plt.title("Non Spam Word Cloud")
            plt.imshow(wordcloud_ham, interpolation="bilinear")
            plt.axis("off")
            plt.show()
            st.pyplot(fig2)
    st.markdown("### 🔎 Feature Exploration")
    df["Characters_Count"] = df["Message"].apply(len)
    df["Words_Count"]=df.apply(lambda row: nltk.word_tokenize(row["Message"]), axis=1).apply(len)
    df["Sentence_Count"]=df.apply(lambda row: nltk.sent_tokenize(row["Message"]), axis=1).apply(len)
    st.write(df.describe().T)
    plt.figure(figsize=(12,8))
    fg = sns.pairplot(data=df, hue="Category",palette=constants.cols) 
    fg.fig.suptitle("Pairplot of DataSet", y=1.08)      
    st.pyplot(fg)
    df = df[(df["Sentence_Count"]<400)]
    plt.figure(figsize=(12,8))
    fg = sns.pairplot(data=df, hue="Category",palette=constants.cols)
    st.markdown("### 🔎 Outliner Removal")
    fg.fig.suptitle("Pairplot of DataSet after removing outlier (Sentence_Count < 400)", y=1.08)  
    st.pyplot(fg)
data_analysis()