from tensorflow.keras.optimizers import SGD  # Stochastic Gradient Descent for optimizations
from tensorflow.keras.layers import Dense, Activation, Dropout  # Features of the Neural Network
from tensorflow.keras.models import Sequential  # provides training features
from nltk.stem import WordNetLemmatizer  # treats words like: work, working, works as work
import random
import json
import pickle
import numpy as np

import nltk

lemmatizer = WordNetLemmatizer()

intents = json.loads(open(r'intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]


# Generate a tuple containing all the intents from intents.json
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)  # Splits sentence into individual words as a list
        words.extend(word_list)
        documents.append((word_list, intent['tag']))  # Ensure word_list appends to the tag category
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# print(documents)

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))  # eliminate duplicates and turn it back into a (now sorted) list

classes = sorted(set(classes))  # sanity check

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

################################### Machine Learning Section ###################################

# Using A Neural Network requires each word to be represented as a numerical value. This will be done using bag of values. Set individual words to
# Either 0 or 1 if its occuring in the particular intent

training = []
output_empty = [0] * len(classes)


# Get all document data in the training list to build the neural network
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        if word in word_patterns:
            bag.append(1)
        else:
            bag.append(0)

    output_row = list(output_empty)  # copy list
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

#  Labels and features to train Neural Network
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Build Sequential Neural Network

model = Sequential()
# 128 neurons, rectified linear unit
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))

model.add(Dropout(0.5))  # Prevent Overfitting (high variance)

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

# softmax scales the results so they all add up to 1
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Feed data 200 times into neural network in a batch size 5
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save it all
model.save('chatbotmodel.h5', hist)
print("Finished")
