# Using a specific, up-to-date Debian release to minimize vulnerabilities
FROM python:3.10-slim-bookworm

# Set the working directory inside the container
WORKDIR /app

# Update package lists and upgrade existing packages to patch vulnerabilities
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache layers
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Load model once at startup and serve requests
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8742"]