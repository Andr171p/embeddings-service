FROM python:3.13-slim

WORKDIR /embeddings-service

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем остальные файлы проекта
COPY . .

ENV MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
ENV PORT=8000

# Предзагрузка модели
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('${MODEL_NAME}')"

EXPOSE ${PORT}

CMD ["python", "main.py"]