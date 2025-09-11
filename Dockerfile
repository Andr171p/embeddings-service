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

# Предзагрузка модели
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('deepvk/USER-bge-m3')"

EXPOSE 8000

CMD ["python", "main.py"]