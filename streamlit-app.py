import pandas as pd
import streamlit as st
import cleantext
from streamlit_shap import st_shap
import shap
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

# Function to load the selected Hugging Face model
@st.cache_resource
def load_model(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
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

# Define sections for input and result
with st.expander('Analyze Text', expanded=True):
    text = st.text_input('Text here: ')
    if text:
        with st.spinner('Calculating...'):
            # Model predictions and SHAP values
            if pipe:
                st.write("Calculating SHAP values and predicting label...")
                explainer = shap.Explainer(pipe)
                shap_values = explainer([text])  # Pass text directly as a list
                predictions = pipe(text)
                prediction = predictions[0][0]['label']
                st.write("SHAP values and prediction calculated.")
                st.write(f"Prediction: {prediction}")
                display_all_labels(predictions)


# Display SHAP values in a separate section
with st.expander('SHAP Values', expanded=True):
    if text:
        with st.spinner('Displaying SHAP values...'):
            if pipe:
                display_shap_values(shap_values, prediction)

# Display SHAP values in a separate section
with st.expander('SHAP Predictions', expanded=True):
    if text:
        with st.spinner('Displaying SHAP Predictions...'):
            if pipe:
                display_all_labels(predictions)