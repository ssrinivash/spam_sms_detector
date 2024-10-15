# Importing necessary libraries
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')

# Loading the dataset 
df = pd.read_csv("spam.csv", encoding="latin-1")

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# changing to lower case nd removing punctuations
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])  # Remove stopwords
    return text

df['message'] = df['message'].apply(preprocess_text)

# Convrting labels to binary (0 for ham, 1 for spam)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})


X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)


vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

bnb = BernoulliNB()
bnb.fit(X_train_vectorized, y_train)

# Predicting on test set
y_pred = bnb.predict(X_test_vectorized)

# model testing
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Use MultinomialNB (another variant of Naive Bayes) for comparison
mnb = MultinomialNB()
mnb.fit(X_train_vectorized, y_train)
y_pred_mnb = mnb.predict(X_test_vectorized)

# Evaluate MultinomialNB model
accuracy_mnb = accuracy_score(y_test, y_pred_mnb)
print(f'\nMultinomialNB Accuracy: {accuracy_mnb * 100:.2f}%')
print("\nMultinomialNB Classification Report:\n", classification_report(y_test, y_pred_mnb))
