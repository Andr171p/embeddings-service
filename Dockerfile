FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Установка uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /embeddings-service

# Копируем файлы конфигурации uv
COPY pyproject.toml uv.lock ./

# Устанавливаем зависимости с помощью uv
RUN uv sync --frozen --no-dev --no-cache

# Копирование остальных файлов проекта
COPY . .

# Создание пользователя приложения и смена владельца рабочей директории
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /embeddings-service

USER appuser

# Создание директории для кеша модели
RUN mkdir -p /home/appuser/.cache/huggingface/hub

ENV MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
ENV PORT=8000

# Предзагрузка модели
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('${MODEL_NAME}')"

EXPOSE ${PORT}

CMD ["python", "main.py"]