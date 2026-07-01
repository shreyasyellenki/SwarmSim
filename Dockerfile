FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY swarmsim/ swarmsim/
COPY weights/ weights/
COPY scripts/ scripts/

EXPOSE 8000
CMD ["uvicorn", "swarmsim.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
