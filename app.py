import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
import re
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

def main(): 
  df = pd.read_csv('Reviews.csv', sep=',', header = 0)
  print(df.head(5))

  #Preparing for pre-process
  data = format_data(df)
  print(data.head(20))

  #PREPROCESSING DATA
  data = preprocess_data(data)
  print(data.head(20))

  #FEATURE EXTRACTION
  feature_extraction(data)




#---PREPROCESSING DATA---
def preprocess_data(data):
  data = to_lowercase(data)
  data = remove_tags_and_links(data)
  data = convert_to_proper_english_syntax(data)
  data = remove_stop_words(data)
  data = lemmatizer(data)
  return data


def to_lowercase(data):
  #lowercase characters and remove html tags
  data["Text"] = data["Text"].apply(lambda x: " ".join(x.lower() for x in str(x).split()))
  return data


def remove_tags_and_links(data):
  #Remove HTML Tags
  data["Text"] = data["Text"].apply(lambda x: BeautifulSoup(x,"html.parser").getText())
  #Remove links: finding http and removing anything character (non-space) that follows it ex: 'http's//www.something.com -> ""
  data["Text"] = data["Text"].apply(lambda x: re.sub(r"http\S+", "", x))
  return data


def convert_to_proper_english_syntax(data):
  #Remove contractions:
  data["Text"] = data["Text"].apply(lambda x: contractions.fix(x))
  #Remove non-alphabetic charcters and tokenize
  data["Text"] = data["Text"].apply(lambda x: " ".join([re.sub("[^a-z]+", "", x) for x in nltk.word_tokenize(x)]))
  #Remove extra spaces
  data["Text"] = data["Text"].apply(lambda x: re.sub(r" +", " ", x))
  return data


def remove_stop_words(data):
  #Remove stop words (frequently used words with no sentmence behind it) such as 'the' , 'a', 'an', 'and', etc.
  stop_words = set(stopwords.words('english'))
  data["Text"] = data["Text"].apply(lambda x: " ".join([ x for x in x.split() if x not in stop_words]))
  return data


def lemmatizer(data):
  #Lemminization: find root meaning behind word, ex: better -> good, running -> run
  lemmatizer = WordNetLemmatizer()
  data["Text"] = data["Text"].apply(lambda x: " ".join([lemmatizer.lemmatize(x) for x in nltk.word_tokenize(x)]))
  return data


def format_data(df):
  #Only grab review text and score
  data = df[["Score", "Text"]]

  #Remove NULL or NONE
  data = data.dropna()
  data = data.reset_index(drop=True)

  #Convert Score to int
  data["Score"] = data["Score"].astype(int)

  data = add_pos_and_neg_labels(data)
  data = randomize_subset(data)

  return data


def add_pos_and_neg_labels(data):
  #Remove neutral reviews
  data = data[data["Score"] != 3]

  #Give label: Yt classification, 1 for positive (>= 4) and 0 for negative (< 4)
  data["Label"] = np.where(data["Score"] >= 4,1,0)
  return data


def randomize_subset(data):
  #Grabbing subset of 100,000 from total dataset: 50% Positive 50% Negative:
  data = data.sample(frac = 1).reset_index(drop = True)
  data1 = data[data['Label'] == 0][:50000]
  data2 = data[data['Label'] == 1][:50000]
  data = data1._append(data2)
  data = data.reset_index(drop = True)

  return data




#---FEATURE EXTRACTION---
def feature_extraction(data):
  #TODO:
  pass



if __name__=="__main__": 
    main() 