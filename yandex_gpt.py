import requests
import numpy as np
from scipy.spatial.distance import cdist

FOLDER_ID = "b1g2phh3iino6u6a244i"
IAM_TOKEN = "t1.9euelZrIxsvHl5qUmZeXiciaypDPye3rnpWajo2Sz8bMzIyOmYmUnZzJj4vl8_c_Wl9M-e91HVgq_d3z938IXUz573UdWCr9zef1656VmpyPxoqLl5mczJfNkZ3PkZ6Y7_zF656VmpyPxoqLl5mczJfNkZ3PkZ6Y.v6lK8SA8Daqg8ZiuuBy2mr1HHT4BXZdzG2ESRbkMJSajzxnrlD_10J7oI0zrTY355gO6k_u-c4FDd-0ZvGasDg"

doc_uri = f"emb://{FOLDER_ID}/text-search-doc/latest"
query_uri = f"emb://{FOLDER_ID}/text-search-query/latest"

embed_url = "https://llm.api.cloud.yandex.net:443/foundationModels/v1/textEmbedding"
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {IAM_TOKEN}", "x-folder-id": f"{FOLDER_ID}"}

doc_texts = [
  """Александр Сергеевич Пушкин (26 мая [6 июня] 1799, Москва — 29 января [10 февраля] 1837, Санкт-Петербург) — русский поэт, драматург и прозаик, заложивший основы русского реалистического направления, литературный критик и теоретик литературы, историк, публицист, журналист.""",
  """Ромашка — род однолетних цветковых растений семейства астровые, или сложноцветные, по современной классификации объединяет около 70 видов невысоких пахучих трав, цветущих с первого года жизни."""
]

query_text = "когда день рождения Пушкина?"

def get_embedding(text: str, text_type: str = "doc") -> np.array:
    query_data = {
        "modelUri": doc_uri if text_type == "doc" else query_uri,
        "text": text,
    }

    return np.array(
        requests.post(embed_url, json=query_data, headers=headers).json()["embedding"]
    )



query_embedding = get_embedding(query_text, text_type="query")
docs_embedding = [get_embedding(doc_text) for doc_text in doc_texts]

# Вычисляем косинусное расстояние
dist = cdist(query_embedding[None, :], docs_embedding, metric="cosine")

# Вычисляем косинусное сходство
sim = 1 - dist

# most similar doc text
print(doc_texts[np.argmax(sim)])
