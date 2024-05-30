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

        if model_options == "nlptown/bert-base-multilingual-uncased-sentiment":
            st.text("Negative: Negative sentiment, Neutral: Neutral sentiment, Positive: Positive sentiment")
            st.text("Negative")
            st_shap(shap.plots.text(shap_values[:, :, "Negative"]))
            st.text("Neutral")
            st_shap(shap.plots.text(shap_values[:, :, "Neutral"]))
            st.text("Positive")
            st_shap(shap.plots.text(shap_values[:, :, "Positive"]))
        elif model_options == "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis":
            st.text("Negative: Negative sentiment, Neutral: Neutral sentiment, Positive: Positive sentiment")
            st.text("Negative")
            st_shap(shap.plots.text(shap_values[:, :, "negative"]))
            st.text("Neutral")
            st_shap(shap.plots.text(shap_values[:, :, "neutral"]))
            st.text("Positive")
            st_shap(shap.plots.text(shap_values[:, :, "positive"]))
        elif model_options == "ElKulako/cryptobert":
            st.text("Bullish: Positive sentiment, Neutral: Neutral sentiment, Bearish: Negative sentiment")
            st.text("Bullish")
            st_shap(shap.plots.text(shap_values[:, :, "Bullish"]))
            st.text("Neutral")
            st_shap(shap.plots.text(shap_values[:, :, "Neutral"]))
            st.text("Bearish")
            st_shap(shap.plots.text(shap_values[:, :, "Bearish"]))

# Analyze Twitter/X Link
with st.expander('Analyze Twitter/X Link'):
    link = st.text_input('Twitter/X Link here: ')

    if link:
        with st.spinner('Calculating...'):
            explainer = shap.Explainer(pipe)
            shap_values = explainer([text])  # Pass text directly as a list

        st.subheader('SHAP Values:')
        st.text("Explanation of SHAP values...")
        st.write(shap_values)  # Display SHAP values
