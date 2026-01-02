FROM python:3.12-slim

WORKDIR /app

COPY .  .

RUN pip install --upgrade pip && \
    pip install mlflow scikit-learn pandas matplotlib

EXPOSE 5000

CMD ["mlflow", "models", "serve", "-m", "artifacts/model", "-h", "0.0.0.0", "-p", "5000"]
