import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from keras.utils import np_utils
from keras.models import Sequential
from keras import layers

plt.style.use('ggplot')

data = pd.read_json('News_Category_Dataset_v2.json', orient='values', lines=True)

# Ograniczenie zakresu czasowego danych do roku 2012
data = data.loc[data['date'].dt.year == 2012]

num_categories = len(set(data.category))

X = data['headline'].values
y = data['category'].values

# Random Forest

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)

vectorizer = CountVectorizer()
vectorizer.fit(X_train)

X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
print('Accuracy:', score)

# Basic DL Model

encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
dummy_y = np_utils.to_categorical(encoded_y)

X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.25, random_state=100)

vectorizer = CountVectorizer()
vectorizer.fit(X_train)

X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

input_dim = X_train.shape[1]
model = Sequential()
model.add(layers.Dense(21, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(num_categories, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=20, verbose=2, validation_data=(X_test, y_test), batch_size=512)

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

plot_history(history)
plt.show()
