FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Установка uv через pip
RUN pip install --no-cache-dir uv

WORKDIR /embeddings-service

# Копируем файлы конфигурации uv
COPY pyproject.toml uv.lock ./

# Устанавливаем зависимости с помощью uv
RUN uv sync --frozen --no-dev --no-cache

# Копируем остальные файлы проекта
COPY . .

# Создание директории для кеша модели
RUN mkdir -p /home/appuser/.cache/huggingface/hub

ENV MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
ENV PORT=8000

# Предзагрузка модели
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('${MODEL_NAME}')"

EXPOSE ${PORT}

CMD ["python", "main.py"]