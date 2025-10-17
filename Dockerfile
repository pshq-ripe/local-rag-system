FROM python:3.12-slim

WORKDIR /app

# Zainstaluj zależności systemowe
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Skopiuj i zainstaluj requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Skopiuj kod aplikacji
COPY rag_server.py .

# Expose port dla FastAPI
EXPOSE 8000

# Uruchom serwer
CMD ["uvicorn", "rag_server:app", "--host", "0.0.0.0", "--port", "8000"]
