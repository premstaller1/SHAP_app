# Sentiment Analysis Streamlit Application
Link to the application: https://sentiment-analysis-stock.streamlit.app/
## Overview
This repository contains a Streamlit application for sentiment analysis using pre-trained models from Hugging Face. The application allows users to input text, upload a CSV file, or provide a tweet link for sentiment analysis. It also provides visual explanations using SHAP values.

## Installation

To set up the application, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repository.git
    cd your-repository
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download necessary NLTK data:
    ```python
    import nltk
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')
    ```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```
## Application Features

### Model Selection
The application allows users to select a pre-trained model from Hugging Face:

1. nlptown/bert-base-multilingual-uncased-sentiment
2. mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
3. ElKulako/cryptobert
4. Custom model (User-specified)

## Input Methods
Users can choose from three input methods for sentiment analysis:

### 1. Text Input:

Enter text directly into the input field.
Displays predictions and SHAP values.

### 2. Upload CSV:

Upload a CSV file containing a column named tweet_text.
Processes and cleans the data.
Provides predictions and visual explanations.

### 3. Tweet Link (Not functional):

Enter a tweet URL.
Fetches the tweet text using Selenium.
Provides sentiment analysis on the fetched text.

### 4. SHAP Values
The application uses SHAP (SHapley Additive exPlanations) to explain the output of the model:

Displays SHAP values for each label.
Provides a bar plot for SHAP values.
Shows SHAP text plots for in-depth analysis.

## Functions
### Model Loading
Loads the selected Hugging Face model for sentiment analysis:

python
@st.cache_resource
def load_model(model_name):
    ...
    
### SHAP Values Display
Displays SHAP values and explanations for the model predictions:

```python
@st.cache_data
def display_shap_values(_shap_values, label):
    ...
```
  
### All Labels Display
Displays all labels and their corresponding scores from the model predictions:

```python
@st.cache_data
def display_all_labels(predictions):
    ...
```
 
### Plot SHAP Values
Plots SHAP values for each label:

```python
@st.cache_data
def plot_shap_values_by_label(_shap_values, labels):
    ...
 ```
   
### Fetch Tweet Text
Fetches the text of a tweet using Selenium:

```python
@st.cache_resource
def fetch_tweet_text(url):
    ...
 ```
   
### Process Tweet
Processes the fetched tweet text for analysis:

```python
@st.cache_data
def process_tweet(url):
    ...
```
    
### Clean Text Functions
Contains functions for cleaning and preprocessing text data:

```python
from preprocessing_script import clean_tweets_column, convert_chat_words, tokenaise, lemmat
```

## Dependencies
```bash
pandas
streamlit
cleantext
streamlit_shap
shap
transformers
matplotlib
numpy
nltk
selenium
webdriver_manager
sklearn
```

## Acknowledgements
Hugging Face for providing pre-trained models.
Streamlit for the interactive application framework.
SHAP for the explanation framework.
For any issues or contributions, please open an issue or submit a pull request on GitHub.
