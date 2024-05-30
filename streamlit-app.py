import pandas as pd
import streamlit as st
import cleantext
import shap
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer

# Load the model and tokenizer
model_name = "ElKulako/cryptobert"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, max_length=64, truncation=True, padding='max_length')

with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        explainer = shap.Explainer(pipe)
        shap_values = explainer([text])  # Pass text directly as a list
        shap_values
        shap.plots.text(shap_values[:, :, "Bullish"])
        shap.plots.bar(shap_values[:, :, "Bearish"].mean(0))