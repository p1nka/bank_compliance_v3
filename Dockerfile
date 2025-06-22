FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gcc g++ build-essential libpq-dev python3-dev libffi-dev libssl-dev libxml2-dev libjpeg-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["sh", "-c", "uvicorn banking_compliance_fastapi:app --host 0.0.0.0 --port ${PORT:-8080}"]