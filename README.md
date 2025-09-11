# HTTP сервер для работы с моделью для векторизации текста


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