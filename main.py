from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import Response
from keras.src.models.model import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import utils
import pandas as pd
import json
import numpy as np


class Command(BaseModel):
    text: str


# Максимальное количество слов
num_words = 10000
# Максимальная длина команды
max_news_len = 10
# Количество классов новостей
nb_classes = 4

train = pd.read_csv('dataset.csv', header=None, names=['class', 'text'])
commands = train['text']

json_file = open('model_new_cnn_3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("best_model_cnn_new_3.keras")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(commands)

app = FastAPI()


@app.post("/class/")  # How to run: "fastapi dev main.py" in terminal
async def get_class(command: Command):
    form_command = command_format(command.text)
    command_arr = [form_command]
    sequences = tokenizer.texts_to_sequences(command_arr)
    x = pad_sequences(sequences, maxlen=max_news_len)
    pred = loaded_model.predict(x[0:1])[0]
    resp = {'text': command.text,
            'showConf': np.float32(pred[0]).item(),
            'allConf': np.float32(pred[1]).item(),
            'logoutConf': np.float32(pred[2]).item(),
            'helpConf': np.float32(pred[3]).item()}
    json_resp = json.dumps(resp)
    print(json_resp)
    return Response(content=json_resp, media_type="application/json")


def command_format(command: str):
    output = ''
    command_arr = command.split()
    for i in range(len(command_arr)):
        if command_arr[i].isdigit():
            if i == len(command_arr) - 1:
                output += '<number>'
            else:
                output += '<number> '
        else:
            if i == len(command_arr) - 1:
                output += command_arr[i]
            else:
                output += command_arr[i] + " "
    return output
