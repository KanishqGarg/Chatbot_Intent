
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
def correct_spelling(text):
    corrected_text = str(TextBlob(text).correct())
    return corrected_text
def vect(feature):
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(feature).toarray()
    return x,vectorizer
