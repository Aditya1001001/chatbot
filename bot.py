import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tflearn
import tensorflow
import json
import random

stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)

dictionary = []
labels = []
docs_X = []
docs_Y = []

# Encountered an error related to missing nltk resource:
# >>> nltk.download('punkt')

for intent in data['intents']:
    for pattern in intent["patterns"]:
        #bringing the words down to their roots
        words = nltk.word_tokenize(pattern)
        dictionary.extend(words)
        docs_X.append(pattern)
        docs_Y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# preprocessing
# getting rid of duplicates and sorting
dictionary = [stemmer.stem(w.lower()) for w in dictionary]
dictionary = sorted(list(set(dictionary)))
labels = sorted(labels)

train = []
output = []

out_empty = [0 for _ in range(len(labels))]

# creating a bag of words
for x, doc in enumerate(docs_X):
    bag = []

    words = [stemmer.stem(w.lower()) for w in doc]
    # one hot encoding of words
    for w in dictionary:
        if w in words:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_Y[x])] = 1

    train.append(bag)
    output.append(output_row)

# converting the data to numpy arrays
train = numpy.array(train)
output = numpy.array(output)

#print(train)