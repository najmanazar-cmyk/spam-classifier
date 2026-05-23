import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('spamfilterdataset.csv')

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit the vectorizer to our dataset and transform it
X = vectorizer.fit_transform(df['Text'])
y = df['Label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Multinomial Naive Bayes classifier
clf = MultinomialNB()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Streamlit app
st.title('Email Spam Classifier')
st.write('Enter an email below to check if it is spam or not.')

# Input text box
email = st.text_area("Enter email content:")

# Classification function
def classify_email(email):
    email_vector = vectorizer.transform([email])
    prediction = clf.predict(email_vector)
    return prediction[0]

# Predict button
if st.button('Classify'):
    if email:
        result = classify_email(email)
        if result == 1:
            st.error("The email is Spam.")
        else:
            st.success("The email is Not Spam.")
    else:
        st.warning("Please enter an email to classify.")
