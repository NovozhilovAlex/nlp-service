from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, MaxPooling1D, Conv1D, GlobalMaxPooling1D, Dropout, LSTM, GRU
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Максимальное количество слов
num_words = 10000
# Максимальная длина команды
max_news_len = 10
# Количество классов новостей
nb_classes = 4

train = pd.read_csv('dataset_2.csv', header=None, names=['class', 'text'])
commands = train['text']
y_train = utils.to_categorical(train['class'] - 1, nb_classes)
print(y_train)

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(commands)
sequences = tokenizer.texts_to_sequences(commands)
x_train = pad_sequences(sequences, maxlen=max_news_len)

print(tokenizer.word_index)

print(commands[1])
print(sequences[1])

model_cnn = Sequential()
model_cnn.add(Embedding(num_words, 16))
model_cnn.add(Conv1D(250, 5, padding='valid', activation='relu'))
model_cnn.add(GlobalMaxPooling1D())
model_cnn.add(Dense(128, activation='relu'))
model_cnn.add(Dense(nb_classes, activation='softmax'))

model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_cnn_save_path = 'best_model_cnn_new_4.keras'
checkpoint_callback_cnn = ModelCheckpoint(model_cnn_save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
history_cnn = model_cnn.fit(x_train, y_train, epochs=150, batch_size=128, validation_split=0.1,
                            callbacks=[checkpoint_callback_cnn])

model_json = model_cnn.to_json()
with open("model_new_cnn_4.json", "w") as json_file:
    json_file.write(model_json)

print('------------------------')
model_cnn.load_weights(model_cnn_save_path)
model_cnn.evaluate(x_train, y_train, verbose=1)






