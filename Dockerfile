# Use the official Python image from Docker Hub
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y python3-distutils

# Set the working directory
WORKDIR /app

# Copy your requirements file and install dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools && \
    pip install -r requirements.txt

# Copy your application code
COPY . .

# Set the entrypoint for the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
