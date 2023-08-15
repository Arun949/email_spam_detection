# email_spam_detection
This code preprocesses email data, builds multiple models (Naive Bayes, SVM, etc.) for spam detection, evaluates their accuracy, plots performance curves, and showcases Cox proportional hazards and PR curve analysis.
This code is a comprehensive example of building and evaluating various machine learning models for email spam detection. It performs the following steps:

1. **Importing Libraries:** The code starts by importing the necessary Python libraries, including data processing, visualization, natural language processing (NLP), and machine learning libraries.

2. **Data Loading and Preprocessing:**
   - The code loads a dataset from a CSV file named "messages.csv" using pandas.
   - It performs initial data exploration, checking for missing values, and handling them by replacing null values with empty strings.
   - It analyzes the distribution of spam and non-spam (ham) labels in the dataset.
   - It calculates and visualizes the distribution of message lengths before and after preprocessing.

3. **Text Preprocessing:**
   - The code applies various text preprocessing techniques to clean the text data. It includes lowercasing, handling email addresses, URLs, currency symbols, and phone numbers, removing punctuation, and dealing with stopwords.
   - It generates WordClouds to visualize the most common words in both spam and non-spam messages.

4. **Feature Extraction:**
   - The text data is transformed into numerical feature vectors using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.
   - The feature vectors are split into training and testing sets.

5. **Model Building and Evaluation:**
   - Several classification algorithms are trained and evaluated for email spam detection:
     - Naive Bayes (MultinomialNB)
     - Support Vector Machine (SVM)
     - Decision Tree Classifier
     - Random Forest Classifier
     - Logistic Regression
     - K-Nearest Neighbors (KNN)
     - Neural Network (MLPClassifier)
     - Gradient Boosting Classifier
     - Gaussian Naive Bayes
     - 1D Convolutional Neural Network (CNN)
   - Each model is trained on the feature vectors and evaluated using accuracy metrics on a testing set.

6. **Model Performance Visualization:**
   - The code visualizes the confusion matrix to assess model performance.
   - It calculates and plots the Precision-Recall curve and displays the average precision score.

7. **Cox Proportional Hazards Curve:**
   - The code demonstrates how to create a Cox proportional hazards model and plot the corresponding curve.

8. **Precision-Recall Curve:**
   - The code generates a Precision-Recall curve for a classification task and calculates the average precision score.

The overall goal of the code is to showcase a comprehensive pipeline for email spam detection, including data preprocessing, feature extraction, model training, evaluation, and performance visualization using a variety of machine learning algorithms. It provides insights into how different models perform on the given dataset and demonstrates how to visualize and analyze their performance.
