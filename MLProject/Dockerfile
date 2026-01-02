FROM python:3.12-slim

WORKDIR /app

# Copy semua file yang diperlukan
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install mlflow scikit-learn pandas matplotlib

# Expose port untuk MLflow model serving (opsional)
EXPOSE 5000

# Command untuk menjalankan model atau serving
CMD ["mlflow", "models", "serve", "-m", "artifacts/model", "-h", "0.0.0.0", "-p", "5000"]
