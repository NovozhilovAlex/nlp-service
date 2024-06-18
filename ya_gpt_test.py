import requests
import numpy as np
from fastapi import FastAPI
from fastapi import Response
import json

from pydantic import BaseModel
from scipy.spatial.distance import cdist

from main import app

FOLDER_ID = "b1g2phh3iino6u6a244i"
IAM_TOKEN = "t1.9euelZrIlZPNz4vHzIqXiYuRyMeQi-3rnpWajo2Sz8bMzIyOmYmUnZzJj4vl9PdHTz9M-e8EaWiw3fT3B348TPnvBGlosM3n9euelZqWipmPj4mSlo2NjI-cycmNle_8xeuelZqWipmPj4mSlo2NjI-cycmNlQ.6JrDEX6gIqpSjN-UnUzbmYk1VuLjtegoC6xERpnnKM-HBB7dNckMQfZ81JLd3WlzBjbE3O7TNuHaOl4_EQkvBw"

model_uri = f"cls://{FOLDER_ID}/yandexgpt/latest"

embed_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/fewShotTextClassification"
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {IAM_TOKEN}", "x-folder-id": f"{FOLDER_ID}"}

text = "Список всех задач"
task_description = "Определи класс команды чат боту"
labels = ["Получение ифнормации о задаче по идентификатору", "Получение списка всех задач пользователя", "Справочная информация", "Выход из системы"]


class Command(BaseModel):
    text: str


@app.post("/class/")  # How to run: "fastapi dev ya_gpt_test.py" in terminal
async def get_class(command: Command):
    res = get_embedding(command.text)

    resp = {'text': command.text,
            'showConf': res[0]["confidence"],
            'allConf': res[1]["confidence"],
            'logoutConf': res[3]["confidence"],
            'helpConf': res[2]["confidence"]}
    json_resp = json.dumps(resp)
    print(json_resp)
    return Response(content=json_resp, media_type="application/json")


def get_embedding(com: str) -> np.array:
    query_data = {
        "modelUri": model_uri,
        "text": com,
        "task_description": task_description,
        "labels": labels
    }

    return np.array(
        requests.post(embed_url, json=query_data, headers=headers).json()["predictions"]
    )


# query_embedding = get_embedding(text)
# print(query_embedding)

