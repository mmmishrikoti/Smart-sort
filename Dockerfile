FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    SORTSMART_DATABASE=/tmp/sortsmart.sqlite3 \
    SORTSMART_UPLOAD_FOLDER=/tmp/sortsmart-uploads

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-space.txt requirements-space.txt
RUN pip install --no-cache-dir -r requirements-space.txt

COPY app.py app.py
COPY templates templates
COPY static static
COPY model/waste_classifier.keras model/waste_classifier.keras
COPY model/labels.json model/labels.json
COPY model/metrics.json model/metrics.json

EXPOSE 7860

CMD ["python", "app.py"]
