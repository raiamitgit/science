FROM mirror.gcr.io/library/python:3.12-slim
COPY af3_batch.py /app/af3_batch.py
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python3", "-B", "/app/af3_batch.py"]
