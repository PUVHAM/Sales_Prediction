# frontend/Dockerfile
FROM python:3.11.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY frontend/ frontend/
COPY app.py .

ENV PYTHONPATH="/app"

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]