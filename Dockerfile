# Use a slim Python image
FROM python:3.10-slim

# Install system audio dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
# We do this first so Docker caches the heavy torch download
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else (your .pth model, app.py, etc.)
COPY . .

EXPOSE 5000

# Start the app
CMD ["python", "app.py"]