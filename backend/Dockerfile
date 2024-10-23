# backend/Dockerfile
FROM python:3.11.9-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY src/ /app/src/
COPY backend/ /app/backend/

RUN mkdir -p /app/data/raw /app/data/processed /app/models

ENV PYTHONPATH="/app"

EXPOSE 8000

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]