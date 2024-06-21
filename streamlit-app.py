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

#For the CSV filme
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

#For twitter Link
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType

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
def display_shap_values(_shap_values, label):
    print("Displaying SHAP values for each label...")
    st.write(f"### SHAP values for {label}")
    # Create a SHAP text plot for the current label
    shap_plot = shap.plots.text(_shap_values[:, :, label])
    st_shap(shap_plot, height=300)

# Function to display all labels and their scores
@st.cache_data
def display_all_labels(predictions):
    st.write("All labels and scores:")
    for label_score in predictions[0]:
        label = label_score['label'].capitalize()
        score = label_score['score']
        st.write(f"{label}: {score:.4f}")

# Function to plot SHAP values by label
@st.cache_data
def plot_shap_values_by_label(_shap_values, labels):
    for label in labels:
        st.write(f"SHAP values for {label}")
        st_shap(shap.plots.bar(shap_values[:, :, label].mean(0), order=shap.Explanation.argsort))


options = Options()
options.add_argument("--disable-gpu")
options.add_argument("--headless")
@st.cache_resource
def get_driver():
    return webdriver.Chrome(
        executable_path=ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install(),
        options=options,
    )

@st.cache_resource
def fetch_tweet_text(url):
    st.title("Test Selenium")
    st.markdown("You should see some random Football match text below in about 21 seconds")

    driver = get_driver()
    driver.get(url)
    time.sleep(10)  # Allow time for the page to load

    # Open the tweet URL
    print("Opening tweet URL")
    driver.get(url)
    time.sleep(10)  # Allow time for the page to load
    print("Page loaded")

    try:
        tweet_text_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'div.css-146c3p1.r-bcqeeo.r-1ttztb7.r-qvutc0.r-37j5jr.r-1inkyih.r-16dba41.r-bnwqim.r-135wba7'))
        )
        tweet_text = tweet_text_element.text
    except Exception as e:
        print(f"Error fetching tweet text: {e}")
        tweet_text = ""
    finally:
        driver.quit()
    
    return tweet_text

@st.cache_data
def process_tweet(url):
    tweet_text = fetch_tweet_text(url)
    print("Original Tweet Text:", tweet_text)
    return tweet_text

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
input_method = st.radio("Choose input method", ("Text Input", "Upload CSV", "Tweet Link"))

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
    with st.expander('Showcasing Shap Values, starting with the highest value', expanded=True):
        if text:
            with st.spinner('Plotting the SHAP label with the highest value...'):
                if pipe:
                    # Extract labels from predictions
                    labels = [pred['label'] for pred in predictions[0]]
                        
                    # Find the label with the highest SHAP value
                    max_shap_label = labels[0]
                    max_shap_value = shap_values[:, :, labels[0]].values.max()
                        
                    for label in labels[1:]:
                        current_max = shap_values[:, :, label].values.max()
                        if current_max > max_shap_value:
                            max_shap_value = current_max
                            max_shap_label = label

                    # Display SHAP values for the label with the highest value
                    display_shap_values(shap_values, max_shap_label)
    with st.expander('Shap Values for all Labels', expanded=True):
        if text:
            with st.spinner('Displaying SHAP values aside from the highest.'):
                if pipe:
                    # Selector for other labels
                    selected_label = st.selectbox("Select label to focus on", labels, index=labels.index(max_shap_label))
                    display_shap_values(shap_values, selected_label)

elif input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Select 100 random rows
        data = data.sample(n=5, random_state=1)
        
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
            with st.spinner('Calculating Sentiment...'):
                predictions = pipe(list(data['text']))

            # Plot the predictions
            st.write("Predictions")
            prediction_labels = [pred[0]['label'] for pred in predictions]
            prediction_scores = [pred[0]['score'] for pred in predictions]
            predictions_df = pd.DataFrame({'original text': data["tweet_text"],'text': data['text'], 'prediction': prediction_labels, 'score': prediction_scores})
            st.write(predictions_df)

            with st.expander("Showcasing Shap Values"):          
                with st.spinner('Calculating SHAP values...'):
                    explainer = shap.Explainer(pipe)
                    shap_values = explainer(list(data['text']))
                    
                    #Display SHAP values
                    st.write("SHAP values and explanations:")
                    unique_labels = list(set(prediction_labels))
                    plot_shap_values_by_label(shap_values, unique_labels)  
            with st.expander("Showcasing Predictions"):          
                top, ax = plt.subplots()
                ax.barh(range(len(prediction_labels)), prediction_scores, align='center')
                ax.set_yticks(range(len(prediction_labels)))
                ax.set_yticklabels(prediction_labels)
                ax.invert_yaxis()  # labels read top-to-bottom
                ax.set_xlabel('Scores')
                ax.set_title('Prediction Scores')
                st.pyplot(top)
                plt.close(top)

                # Plot the ratio of prediction labels
                st.write("Ratio of Prediction Labels")
                label_counts = pd.Series(prediction_labels).value_counts()
                bottom, ax = plt.subplots()
                ax.bar(label_counts.index, label_counts.values)
                ax.set_xlabel('Labels')
                ax.set_ylabel('Count')
                ax.set_title('Ratio of Prediction Labels')
                st.pyplot(bottom) 
                plt.close(bottom)
        
        else:
            st.error('The CSV file must contain a "tweet_text" column.')

elif input_method == "Tweet Link":
    tweet_url = st.text_input('Paste the tweet link here:')
    if tweet_url:
        with st.spinner('Fetching tweet text...'):
            tweet_text = process_tweet(tweet_url)
        st.write(f"Fetched Tweet Text: {tweet_text}")