# Sentiment Analysis Project
# Description
This is a Linear Support Vector Classification (LinearSVC) model that utilizes Term Frequency-Inverse Document Frequency (TF-IDF) vectorization for feature extraction. This model is trained for the purpose of performing sentiment analysis on a product review and determining if the review is positive (1) or negative (0).
# Training the model
To train and evaluate the model, first make sure to extract the `condense_Reviews` file from `condense_Reviews.zip` and put it in the root folder. Then, rename the file to `Reviews.csv`.
Lastly, run the file `app.py`.
This will save the model and IDF-vector under the folder `SentimentANalysisModel` and output a classification report for the model which includes its accuracy.

# Testing the model
While simply running `app.py` already tests the model and even outputs a report for its performance, the model can also be tested with user input.
To do so, only after having already ran `app.py` before, run the file `test_model.py`. The program will ask for you to input a review so that the model can predict if it is positive or negative.
Type out the review you want to test and press enter. The output will be either "1 (Positive)" or "0 (Negative)".
