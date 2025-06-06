import os
from utils.chatbot import chatbot
import joblib
import json
from utils.preprocessing import correct_spelling
with open('Intents.json', 'r') as file:
  intents = json.load(file)
model = joblib.load('model/model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')
while True:
    text=input(" Hey this is your chatbot!! what do you want to say? \n")
    text=correct_spelling(text)
    tag,output= chatbot(model,intents,text,vectorizer)
    print(output)
    if text=='exit' or tag=='goodbye':
        break