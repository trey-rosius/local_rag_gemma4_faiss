FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for FAISS and PDF processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install project dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment variables for DBOS
ENV DBOS_DB_HOSTNAME=db
ENV DBOS_DB_PORT=5432
ENV DBOS_DB_USERNAME=postgres
ENV DBOS_DB_PASSWORD=password
ENV DBOS_DB_NAME=dbos

# Default command: run the ingestion workflow
CMD ["python", "durable_ingest.py"]
