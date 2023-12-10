from bs4 import BeautifulSoup
import re
import contractions
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# ---PREPROCESSING DATA---
def preprocess_data(data):
  data = to_lowercase(data)
  data = remove_tags_and_links(data)
  data = convert_to_proper_english_syntax(data)
  data = remove_stop_words(data)
  data = lemmatizer(data)

  return data


def to_lowercase(data):
  # lowercase characters and remove html tags
  data = " ".join(x.lower() for x in str(data).split())

  return data


def remove_tags_and_links(data):
  # Remove HTML Tags
  data = BeautifulSoup(data,"html.parser").getText()
  # Remove links: finding http and removing anything character (non-space) that follows it ex: 'http's//www.something.com -> ""
  data = re.sub(r"http\S+", "", data)

  return data


def convert_to_proper_english_syntax(data):
  # Remove contractions:
  data = contractions.fix(data)
  # Remove non-alphabetic charcters and tokenize
  data = " ".join([re.sub("[^a-z]+", "", x) for x in nltk.word_tokenize(data)])
  # Remove extra spaces
  data = re.sub(r" +", " ", data)

  return data


def remove_stop_words(data):
  # Remove stop words (frequently used words with no sentmence behind it) such as 'the' , 'a', 'an', 'and', etc.
  stop_words = set(stopwords.words('english'))
  data = " ".join([x for x in data.split() if x not in stop_words])

  return data


def lemmatizer(data):
  # Lemminization: find root meaning behind word, ex: better -> good, running -> run
  lemmatizer = WordNetLemmatizer()
  data = " ".join([lemmatizer.lemmatize(x) for x in nltk.word_tokenize(data)])

  return data

def main(): 
    model = pickle.load(open("SentimentAnalysisModel/sentiment_analysis_model.pkl", 'rb'))
    model_input = input("Enter review: ")
    model_input_preprocessed = preprocess_data(model_input)
    X_test = pd.Series([model_input_preprocessed])

     # Initialize a TF-IDF vectorizer
    tf_vectorizer = pickle.load(open("SentimentAnalysisModel/idfvector.pkl", 'rb'))

    # Transform the input into TF-IDF features
    tf_x_test = tf_vectorizer.transform(X_test)

    # Predict the labels for the input
    model_output = model.predict(tf_x_test)[0]
    print('{} ({})'.format(model_output, 'Negative' if model_output == 0 else 'Positive'))

if __name__=="__main__": 
  main() 