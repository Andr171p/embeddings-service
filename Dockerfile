FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /embeddings-service

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Создание директории для кеша модели
RUN mkdir -p /home/appuser/.cache/huggingface/hub && \
    chown -R appuser:appuser /home/appuser/.cache

ENV MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
ENV PORT=8000

RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('${MODEL_NAME}')"

EXPOSE ${PORT}

CMD ["/bin/bash", "-c", "python main.py"]
