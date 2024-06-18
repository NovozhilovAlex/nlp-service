from keras.src.models.model import model_from_json
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

train = pd.read_csv('dataset.csv', header=None, names=['class', 'text'])
commands = train['text']
y_train = utils.to_categorical(train['class'] - 1, nb_classes)
command = 'Выйти из аккаунта'

tokenizer = Tokenizer(num_words=num_words)
command_arr = [command]
tokenizer.fit_on_texts(commands)
sequences = tokenizer.texts_to_sequences(commands)
print(command_arr[0])
print(sequences[0])
x_train = pad_sequences(sequences, maxlen=max_news_len)

# model_gru = Sequential()
# model_gru.add(Embedding(num_words, 32, input_length=max_news_len))
# model_gru.add(GRU(16))
# model_gru.add(Dense(2, activation='softmax'))
#
# model_gru.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model_gru_save_path = 'best_model_gru.keras'
# checkpoint_callback_gru = ModelCheckpoint(model_gru_save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
# history_gru = model_gru.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1,
#                             callbacks=[checkpoint_callback_gru])

json_file = open('model_new_cnn_3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("best_model_cnn_new_3.keras")

loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(x_train, y_train, verbose=0)
print(loaded_model.predict(x_train[0:1])[0])
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))