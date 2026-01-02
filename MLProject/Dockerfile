FROM python:3.12-slim

WORKDIR /app

# Copy all files from MLProject directory
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir mlflow scikit-learn pandas matplotlib

# Expose port for MLflow model serving (optional)
EXPOSE 5000

# Default command - you can override this
CMD ["python", "modelling.py"]
