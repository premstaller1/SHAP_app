import pandas as pd
import streamlit as st
import cleantext
from streamlit_shap import st_shap
import shap
import transformers
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
from preprocessing_script import clean_tweets_column, convert_chat_words, tokenaise, lemmatize_text
from sklearn.feature_extraction.text import CountVectorizer
import string
import emoji
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

# Function to load the selected Hugging Face model
@st.cache_resource
def load_model(model_name):
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
        pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, max_length=64, truncation=True, top_k=None, padding='max_length')
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to display SHAP values and explanations based on model type
@st.cache_data
def display_shap_values(_shap_values, prediction):
    print("Displaying SHAP values...")
        # Display SHAP values
    st_shap(shap.plots.text(shap_values))

# Function to display all labels and their scores
@st.cache_data
def display_all_labels(predictions):
    st.write("All labels and scores:")
    for label_score in predictions[0]:
        label = label_score['label'].capitalize()
        score = label_score['score']
        st.write(f"{label}: {score:.4f}")


# Streamlit UI
st.header('Sentiment Analysis')

# Model selection
model_options = [
    "nlptown/bert-base-multilingual-uncased-sentiment",
    "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    "ElKulako/cryptobert",
    "Own Model"
]

selected_model = st.selectbox("Select Hugging Face model", model_options)

# Check if "Own Model" is selected
if selected_model == "Own Model":
    custom_model_name = st.text_input("Enter custom Hugging Face model name (optional)")
    if custom_model_name:
        pipe = load_model(custom_model_name)
else:
    pipe = load_model(selected_model)

# Step 2: Choose input method
input_method = st.radio("Choose input method", ("Text Input", "Upload CSV"))

# Step 3: Input data analysis or uploaded data analysis
if input_method == "Text Input":
    with st.expander('Analyze Text', expanded=True):
        text = st.text_input('Text here: ')
        if text:
            with st.spinner('Calculating...'):
                # Model predictions and SHAP values
                if pipe:
                    explainer = shap.Explainer(pipe)
                    shap_values = explainer([text])  # Pass text directly as a list
                    predictions = pipe(text)
                    prediction = predictions[0][0]['label']
                    st.write(f"Prediction: {prediction}")
                    display_all_labels(predictions)

    # Display SHAP values in a separate section
    with st.expander('SHAP Values', expanded=True):
        if text:
            with st.spinner('Displaying SHAP values...'):
                if pipe:
                    display_shap_values(shap_values)

elif input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Limit the data to 100 rows
        data = data.head(100)
        
        # Select only the "tweet_text" column
        if 'tweet_text' in data.columns:
            data = data[["tweet_text"]]

            # Clean the tweets column
            clean_tweets_column(data, 'tweet_text', 'text')
            
            # Convert chat words
            data['text'] = data['text'].apply(convert_chat_words)
            
            # Tokenize the cleaned text
            data['token_text'] = data['text'].apply(tokenaise)
            
            # Lemmatize the tokens
            data['token_text'] = data['token_text'].apply(lemmatize_text)
            
            st.write('Cleaned Data')
            st.write(data)
            
            # Allow download of cleaned data
            cleaned_data_csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download cleaned data as CSV",
                data=cleaned_data_csv,
                file_name='cleaned_data.csv',
                mime='text/csv',
            )
            # Run the pipeline and get predictions
            st.spinner("Running sentiment analysis...")
            predictions = pipe(list(data['text']))

            # Plot the predictions
            st.write("Predictions")
            prediction_labels = [pred[0]['label'] for pred in predictions]
            prediction_scores = [pred[0]['score'] for pred in predictions]

            fig, ax = plt.subplots()
            ax.barh(range(len(prediction_labels)), prediction_scores, align='center')
            ax.set_yticks(range(len(prediction_labels)))
            ax.set_yticklabels(prediction_labels)
            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_xlabel('Scores')
            ax.set_title('Prediction Scores')
            st.pyplot(fig)

            # Plot the ratio of prediction labels
            st.write("Ratio of Prediction Labels")
            label_counts = pd.Series(prediction_labels).value_counts()
            fig, ax = plt.subplots()
            ax.bar(label_counts.index, label_counts.values)
            ax.set_xlabel('Labels')
            ax.set_ylabel('Count')
            ax.set_title('Ratio of Prediction Labels')
            st.pyplot(fig)
            
            # Get SHAP values
            st.spinner('Calculating SHAP values...')
            explainer = shap.Explainer(pipe)
            shap_values = explainer(list(data['text']))
            
           # Display SHAP values
            st.spinner('Creation plot for SHAP values...')
            st.write("SHAP values and explanations:")
            st_shap(shap.plots.waterfall(shap_values[0]), height=300)
            st_shap(shap.plots.beeswarm(shap_values), height=300)
            
            explainer = shap.TreeExplainer(pipe.model)
            shap_values_tree = explainer.shap_values(data['text'])
            X_display = data['text']
            
            st_shap(shap.force_plot(explainer.expected_value, shap_values_tree[0,:], X_display.iloc[0]), height=200, width=1000)
            st_shap(shap.force_plot(explainer.expected_value, shap_values_tree[:1000,:], X_display.iloc[:1000]), height=400, width=1000)
        else:
            st.error('The CSV file must contain a "tweet_text" column.')