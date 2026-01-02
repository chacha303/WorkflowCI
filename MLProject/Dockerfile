FROM python:3.12-slim

WORKDIR /app

# Copy requirements dan install dependencies
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file project
COPY . .

# Command default (bisa disesuaikan)
CMD ["python", "modelling.py"]
