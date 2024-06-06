import pandas as pd
import streamlit as st
import cleantext
from streamlit_shap import st_shap
import shap
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

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

# Function to display SHAP values and explanations based on model type
def display_shap_values(model_name, shap_values, prediction):
    st_shap(shap.plots.text(shap_values[:, :, "prediction"]))

# Streamlit UI
st.header('Sentiment Analysis')

# Model selection
model_options = [
    "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    "ElKulako/cryptobert",
    "Own Model"
]

selected_model1 = st.selectbox("Select Hugging Face model 1", model_options)
selected_model2 = st.selectbox("Select Hugging Face model 2", model_options)

# Check if "Own Model" is selected for model 1
if selected_model1 == "Own Model":
    custom_model_name1 = st.text_input("Enter custom Hugging Face model name for Model 1 (optional)", key="custom_model1")
    if custom_model_name1:
        pipe1 = load_model(custom_model_name1)
else:
    pipe1 = load_model(selected_model1)

# Check if "Own Model" is selected for model 2
if selected_model2 == "Own Model":
    custom_model_name2 = st.text_input("Enter custom Hugging Face model name for Model 2 (optional)", key="custom_model2")
    if custom_model_name2:
        pipe2 = load_model(custom_model_name2)
else:
    pipe2 = load_model(selected_model2)

with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        with st.spinner('Calculating...'):
            # Model 1 predictions and SHAP values
            if pipe1:
                st.text("Model 1: Calculating SHAP values and predicting label...")
                explainer1 = shap.Explainer(pipe1)
                shap_values1 = explainer1([text])  # Pass text directly as a list
                prediction1 = pipe1(text)[0]['label']
                st.text("Model 1: SHAP values and prediction calculated.")
                st.write(f"Model 1 Prediction: {prediction1}")
                display_shap_values(selected_model1 if selected_model1 != "Own Model" else custom_model_name1, shap_values1, prediction1)

            # Model 2 predictions and SHAP values
            if pipe2:
                st.text("Model 2: Calculating SHAP values and predicting label...")
                explainer2 = shap.Explainer(pipe2)
                shap_values2 = explainer2([text])  # Pass text directly as a list
                prediction2 = pipe2(text)[0]['label']
                st.text("Model 2: SHAP values and prediction calculated.")
                st.write(f"Model 2 Prediction: {prediction2}")
                display_shap_values(selected_model2 if selected_model2 != "Own Model" else custom_model_name2, shap_values1, prediction2)