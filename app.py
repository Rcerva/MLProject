import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
import re
import contractions
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

import pickle

def main(): 
  df = pd.read_csv('Reviews.csv', sep=',', header = 0)
  print()
  print(df.head(5))

  # Preparing for pre-process
  print()
  print("Preparing Pre-Process")
  data = format_data(df)
  print(data.head(20))

  print()
  print("Pre-Processing")
  # PREPROCESSING DATA
  data = preprocess_data(data)
  print(data.head(20))

  # Initialize a Linear Support Vector Classification (LinearSVC) model
  model = LinearSVC(random_state = 0)
  sigmoid_model = LogisticRegression(solver='liblinear')
  
  print()
  print("Feature Extraction")
  # FEATURE EXTRACTION
  report = feature_extraction(data, model)
  test_model = feature_extraction(data, sigmoid_model)

  # Print the classification report
  print()
  print("***REPORT***\n")
  print('1) Results for Negative reviews\n')
  print('      SVM Model\t\t\t\t     Logistic Regression Model')
  print(f" - Precision: {report['0']['precision']}\t - Precision: {test_model['0']['precision']}\n - Recall: {report['0']['recall']}\t\t - Recall: {test_model['0']['recall']}\n\n")
  print('2) Results for Positive reviews\n')
  print('      SVM Model\t\t\t\t     Logistic Regression Model')
  print(f" - Precision: {report['1']['precision']}\t - Precision: {test_model['1']['precision']}\n - Recall: {report['1']['recall']}\t\t - Recall: {test_model['1']['recall']}\n")

  print(f"3) Logistic vs SVM model")
  print(f"Logistic regression model's accuracy: {test_model['accuracy']}")
  print(f"SVM's model's accuracy: {report['accuracy']}\n")


  save_model(model)
  print('End of program...')



def save_model(model):
  # Save model as pickle file
  sentiment_analysis_model = "sentiment_analysis_model.pkl"  

  # Write Pickle file 
  with open("SentimentAnalysisModel/" + sentiment_analysis_model, 'wb') as file:  
    pickle.dump(model, file)
  
  print("Model Saved!")

def save_idfvector(idfvector):
  # Save model as pickle file
  sentiment_analysis_idfvector = "idfvector.pkl"  

  # Write Pickle file 
  with open("SentimentAnalysisModel/" + sentiment_analysis_idfvector, 'wb') as file:  
    pickle.dump(idfvector, file)
  
  print("IDF-vector Saved!")


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
  data["Text"] = data["Text"].apply(lambda x: " ".join(x.lower() for x in str(x).split()))

  return data


def remove_tags_and_links(data):
  # Remove HTML Tags
  data["Text"] = data["Text"].apply(lambda x: BeautifulSoup(x,"html.parser").getText())
  # Remove links: finding http and removing anything character (non-space) that follows it ex: 'http's//www.something.com -> ""
  data["Text"] = data["Text"].apply(lambda x: re.sub(r"http\S+", "", x))

  return data


def convert_to_proper_english_syntax(data):
  # Remove contractions:
  data["Text"] = data["Text"].apply(lambda x: contractions.fix(x))
  # Remove non-alphabetic charcters and tokenize
  data["Text"] = data["Text"].apply(lambda x: " ".join([re.sub("[^a-z]+", "", x) for x in nltk.word_tokenize(x)]))
  # Remove extra spaces
  data["Text"] = data["Text"].apply(lambda x: re.sub(r" +", " ", x))

  return data


def remove_stop_words(data):
  # Remove stop words (frequently used words with no sentmence behind it) such as 'the' , 'a', 'an', 'and', etc.
  stop_words = set(stopwords.words('english'))
  data["Text"] = data["Text"].apply(lambda x: " ".join([ x for x in x.split() if x not in stop_words]))

  return data


def lemmatizer(data):
  # Lemminization: find root meaning behind word, ex: better -> good, running -> run
  lemmatizer = WordNetLemmatizer()
  data["Text"] = data["Text"].apply(lambda x: " ".join([lemmatizer.lemmatize(x) for x in nltk.word_tokenize(x)]))

  return data


def format_data(df):
  # Only grab review text and score
  data = df[["Score", "Text"]]

  # Remove NULL or NONE
  data = data.dropna()
  data = data.reset_index(drop=True)

  # Convert Score to int
  data["Score"] = data["Score"].astype(int)

  data = add_pos_and_neg_labels(data)
  data = obtain_subset(data)

  return data


def add_pos_and_neg_labels(data):
  # Remove neutral reviews
  data = data[data["Score"] != 3]

  # Give label: Yt classification, 1 for positive (>= 4) and 0 for negative (< 4)
  data["Label"] = np.where(data["Score"] >= 4,1,0)

  return data


def obtain_subset(data):
  # Grabbing random subset of 100,000 from total dataset: 50% Positive 50% Negative:
  data = data.sample(frac = 1).reset_index(drop = True)

  # 50,000 negative
  data1 = data[data['Label'] == 0][:50000]

  # 50,000 positive
  data2 = data[data['Label'] == 1][:50000]

  # Combine the negatives and positives
  data = data1._append(data2)
  data = data.reset_index(drop = True)

  return data




# ---FEATURE EXTRACTION---
def feature_extraction(data, model):
  # Split the data into training and testing sets
  X_train, X_test, Y_train, Y_test = train_test_split(data["Text"], data["Label"], test_size = 0.25, random_state = 30)

  # Initialize a TF-IDF vectorizer
  tf_vectorizer = TfidfVectorizer()

  # Transform the training data into TF-IDF features
  tf_x_train = tf_vectorizer.fit_transform(X_train)

  #Save IDF vector for use in predictions
  save_idfvector(tf_vectorizer)

  # Transform the testing data using the same vectorizer
  tf_x_test = tf_vectorizer.transform(X_test)

  # Train the model using the TF-IDF features and corresponding labels
  model.fit(tf_x_train, Y_train)

  # Predict the labels for the testing data
  y_test_pred = model.predict(tf_x_test)

  # Generate a classification report to evaluate the model's performance
  # The report includes precision, recall, F1-score, and support for each class
  report = classification_report(Y_test, y_test_pred, output_dict = True)

  return report


if __name__=="__main__": 
  main() 