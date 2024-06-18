from sklearn.feature_extraction.text import CountVectorizer
import nltk 
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import emoji
import re

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
from nltk.corpus import stopwords

def clean_tweets_column(df, column_name, cleaned_column):
    # Function to convert emojis to words using emoji library mapping
    def convert_emojis_to_words(tweet):
        converted_text = emoji.demojize(tweet)
        return converted_text
    
    def clean_tweet(tweet):
        # Convert to string if it's not already
        tweet = str(tweet)
        # Remove hyperlinks
        tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
        # Remove user @ references and '#' from hashtags
        tweet = re.sub(r'\@\w+|\#','', tweet)
        # Remove emojis
        tweet = re.sub(r'[^\x00-\x7F]+', '', tweet)
        # Remove new lines
        tweet = tweet.replace('\n', ' ')
        # Remove extra spaces
        tweet = re.sub(r'\s+', ' ', tweet).strip()
        # Convert to lowercase
        tweet = tweet.lower()
        # Remove special characters, numbers, and punctuations
        tweet = "".join([char for char in tweet if char not in string.punctuation])
        tweet = re.sub('[0-9]+', '', tweet)
        return tweet

    def remove_stopwords(tweet):
        stop_words = set(stopwords.words('english'))
        tweet_tokens = tweet.split()
        filtered_tweet = [word for word in tweet_tokens if word not in stop_words]
        return ' '.join(filtered_tweet)

    # Apply the function to the 'text_cleaned' column in the DataFrame
    df[cleaned_column] = df[column_name].apply(convert_emojis_to_words)

    # Apply the clean_tweet function to the specified column
    df[cleaned_column] = df[column_name].astype(str).apply(clean_tweet)
    # Remove stop words
    df[cleaned_column] = df[cleaned_column].apply(remove_stopwords)

def convert_chat_words(text):
    #Convert possible chat words into full words:
    #convert chatwords 
    chat_words_dict = {
    "imo": "in my opinion",
    "cyaa": "see you",
    "idk": "I don't know",
    "rn": "right now",
    "afaik": "as far as I know",
    "brb": "be right back",
    "btw": "by the way",
    "cya": "see you",
    "dm": "direct message",
    "ffs": "for f*ck's sake",
    "fml": "f*ck my life",
    "ftw": "for the win",
    "hmu": "hit me up",
    "icymi": "in case you missed it",
    "idc": "I don't care",
    "idgaf": "I don't give a f*ck",
    "idts": "I don't think so",
    "iirc": "if I recall correctly",
    "ikr": "I know, right?",
    "ily": "I love you",
    "imho": "in my humble opinion"
    }

    words = text.split()

    converted_words = []

    for word in words:
        if word.lower() in chat_words_dict:
            converted_words.append(chat_words_dict[word.lower()])
        else:
            converted_words.append(word)
    converted_text = " ".join(converted_words)
    return converted_text

# Function to apply tokens and stop word removal to a text
def tokenaise(text):
    tokens = word_tokenize(text)  # Tokenize the text
    stop_words = set(stopwords.words('english'))  # Get the stop words
    tokens = [word for word in tokens if word not in stop_words]  # Remove stop words
    return tokens

# Function to perform Lemmatization on a text
def lemmatize_text(tweet):
    # Create an instance of WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    # POS tag mapping dictionary
    wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}
    # Get the POS tags for the words
    pos_tags = nltk.pos_tag(tweet)
    
    # Perform Lemmatization
    lemmatized_words = []
    for word, tag in pos_tags:
        # Map the POS tag to WordNet POS tag
        pos = wordnet_map.get(tag[0].upper(), wordnet.NOUN)
        # Lemmatize the word with the appropriate POS tag
        lemmatized_word = lemmatizer.lemmatize(word, pos=pos)
        # Add the lemmatized word to the list
        lemmatized_words.append(lemmatized_word)
    
    return lemmatized_words