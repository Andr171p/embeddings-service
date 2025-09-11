import uvicorn

from embeddings_service.app import app
from embeddings_service.depends import settings

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.port, log_level="info")  # noqa: S104
