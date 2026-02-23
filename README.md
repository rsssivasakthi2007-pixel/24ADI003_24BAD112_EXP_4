Scenario 1 – Multinomial Naïve Bayes (SMS Spam Classification)

Dataset (Kaggle – Public) https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

This scenario focuses on classifying SMS messages as Spam or Ham using the SMS Spam Collection Dataset. The raw text messages are preprocessed through lowercase conversion, punctuation removal, and optional stopword filtering. The cleaned text is transformed into numerical features using Count Vectorization or TF-IDF. A Multinomial Naïve Bayes classifier is trained on the processed data and evaluated using Accuracy, Precision, Recall, F1-score, and a Confusion Matrix. Feature importance analysis and Laplace smoothing are also performed to understand the impact of word probabilities on spam detection.

Scenario 2 – Gaussian Naïve Bayes (Iris Classification)

Dataset (Public / Standard Dataset)

Iris Dataset (sklearn)
This scenario uses the Iris Dataset to classify flower species based on four numerical features: sepal length, sepal width, petal length, and petal width. After data inspection and feature scaling, the dataset is split into training and testing sets. A Gaussian Naïve Bayes classifier is trained to predict flower species and evaluated using Accuracy, Precision, Recall, F1-score, and a Confusion Matrix. Decision boundary visualization and probability analysis help in understanding the model’s classification performance.

# 24ADI003_24BAD112_EXP_4
