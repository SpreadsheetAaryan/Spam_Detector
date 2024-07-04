# Spam Detector Project

This project implements a spam detector using a machine learning pipeline in Python. The pipeline processes email data, converts it to a format suitable for machine learning, and trains a logistic regression model to classify emails as spam or not spam.

## Data

The project expects the email dataset to be provided as two lists: `X` containing the email contents and `y` containing the corresponding labels.

## Model

The machine learning pipeline consists of the following steps:
1. `EmailToWordCounterTransformer`: Transforms emails into word count vectors.
2. `WordCounterToVectorTransformer`: Converts word count vectors into numerical vectors suitable for machine learning.
3. `LogisticRegression`: Trains a logistic regression model on the processed data.

## Evaluation

After training, the model is evaluated using accuracy as the metric. The evaluation script outputs the accuracy of the model on the test dataset.

## Process of Building

Building the spam detector project was a comprehensive and enlightening experience. The journey began with the crucial step of data preprocessing, which is often the most time-consuming part of any machine learning project. The raw email data needed to be transformed into a format suitable for training a machine learning model. This involved tokenizing the email content, converting the text into numerical features, and normalizing these features. I spent considerable time studying how to effectively preprocess text data, which included learning about various techniques like word count vectorization and Term Frequency-Inverse Document Frequency (TF-IDF). To streamline this process, I had to dive deep into creating custom transformers and integrating them into a scikit-learn pipeline, ensuring that each step of the preprocessing was systematically and efficiently applied.

Initially, I chose to implement a Naive Bayes classifier due to its simplicity and efficiency in text classification tasks. However, despite its theoretical advantages, the model did not yield satisfactory accuracy. This discrepancy led me to revisit the preprocessing steps and the assumptions made by the Naive Bayes algorithm about feature independence, which might not hold true for the email data. I decided to switch to Logistic Regression, a more complex and widely-used algorithm for binary classification. Setting the max_iter parameter to 1000 ensured that the model had sufficient iterations to converge to an optimal solution. This change significantly improved the accuracy, and the model's performance was much more in line with the expected results. Through this process, I gained a deeper understanding of model selection, hyperparameter tuning, and the importance of iterative testing and validation in machine learning.