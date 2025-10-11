# HTTP сервер для работы с моделью для векторизации текста

## Установка

### Langchain версия
```shell
uv add "embeddings-service[langchain] @ git+https://github.com/Andr171p/embeddings-service.git"
```

### Серверная версия
```shell
uv add "embeddings-service[server] @ git+https://github.com/Andr171p/embeddings-service.git"
```


## Интеграция с LangChain
```python
from embeddings_service.langchain import RemoteHTTPEmbeddings

embeddings = RemoteHTTPEmbeddings(
    base_url="http://localhost:8000",
    normalize_embeddings=True,
    batch_size=32
)

text = "Hello world"

vectors = embeddings.embed_documents([text])
```