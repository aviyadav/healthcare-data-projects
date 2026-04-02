"""
Project entry point.

Development:
    python main.py             # runs uvicorn directly

Production:
    gunicorn -c gunicorn.conf.py app.main:app
"""
import uvicorn
from app.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,           # hot-reload in development
        log_level="info",
    )
