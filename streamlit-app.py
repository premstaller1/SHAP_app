import pandas as pd
import streamlit as st
import cleantext
from streamlit_shap import st_shap
import shap
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer

# Function to load the selected Hugging Face model
def load_model(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, max_length=64, truncation=True, padding='max_length')
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

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

with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        with st.spinner('Calculating...'):
            explainer = shap.Explainer(pipe)
            shap_values = explainer([text])  # Pass text directly as a list

        st.subheader('SHAP Values:')
        st.text("Explanation of SHAP values...")
        shap_values

        if selected_model == "nlptown/bert-base-multilingual-uncased-sentiment":
            st.text("Negative: Negative sentiment, Neutral: Neutral sentiment, Positive: Positive sentiment")
            st.text("Negative")
            st_shap(shap.plots.text(shap_values[:, :, "Negative"]))
            st.text("Neutral")
            st_shap(shap.plots.text(shap_values[:, :, "Neutral"]))
            st.text("Positive")
            st_shap(shap.plots.text(shap_values[:, :, "Positive"]))
        elif selected_model == "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis":
            st.text("Negative: Negative sentiment, Neutral: Neutral sentiment, Positive: Positive sentiment")
            st.text("Negative")
            st_shap(shap.plots.text(shap_values[:, :, "negative"]))
            st.text("Neutral")
            st_shap(shap.plots.text(shap_values[:, :, "neutral"]))
            st.text("Positive")
            st_shap(shap.plots.text(shap_values[:, :, "positive"]))
        elif selected_model == "ElKulako/cryptobert":
            st.text("Bullish: Positive sentiment, Neutral: Neutral sentiment, Bearish: Negative sentiment")
            st.text("Bullish")
            st_shap(shap.plots.text(shap_values[:, :, "Bullish"]))
            st.text("Neutral")
            st_shap(shap.plots.text(shap_values[:, :, "Neutral"]))
            st.text("Bearish")
            st_shap(shap.plots.text(shap_values[:, :, "Bearish"]))

import requests
from bs4 import BeautifulSoup

def extract_text_from_twitter_post(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Find the element containing the text content of the tweet
            tweet_text_element = soup.find('div', {'class': 'css-901oao r-hkyrab r-1qd0xha r-a023e6 r-16dba41 r-ad9z0x r-bcqeeo r-bnwqim r-qvutc0'})
            if tweet_text_element:
                return tweet_text_element.text.strip()
            else:
                return "Text not found in Twitter post."
        else:
            return "Failed to retrieve Twitter post. Status code: " + str(response.status_code)
    except Exception as e:
        return "Error: " + str(e)

# Streamlit UI
st.header('Sentiment Analysis')

# Analyze Twitter/X Link
with st.expander('Analyze Twitter/X Link'):
    link = st.text_input('Twitter/X Link here: ')

    if link:
        with st.spinner('Crawling Twitter post...'):
            tweet_text = extract_text_from_twitter_post(link)
            if "Error" not in tweet_text and "Failed" not in tweet_text:
                st.write("Extracted text from Twitter post:")
                st.write(tweet_text)
            else:
                st.write(tweet_text)
