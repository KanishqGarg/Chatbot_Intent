import os
import nltk
import random
from sklearn.linear_model import LogisticRegression
from utils.preprocessing import vect
from utils.model_utils import build_model
import json
import pandas as pd
import joblib

with open('Intents.json', 'r') as file:
  intents = json.load(file)
tags = []
patterns = []

for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)
x,vectorizer=vect(patterns)
model=build_model(x,tags)
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')

