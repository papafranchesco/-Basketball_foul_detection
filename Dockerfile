# ---------- Dockerfile ----------
# Базовый образ с CUDA 12 и Ubuntu 22 (подходит под Cloud Run GPU)
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Без взаимодействия
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Системные пакеты: ffmpeg + python
RUN apt-get update && \
    apt-get install -y ffmpeg python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Скопировать код
COPY . .

# Установить Python-зависимости
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Открытый порт (FastAPI по умолч. 8080)
EXPOSE 8080

# Команда запуска (предположим, у вас есть server/app.py c FastAPI)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8080"]
# --------------------------------
