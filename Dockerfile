# Use the official Python image from the Docker Hub
FROM python:3.12.0

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt into the container
COPY requirements.txt /app/

RUN apt-get update && apt-get install -y build-essential libatlas-base-dev libblas-dev liblapack-dev
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools to avoid version-related issues
RUN pip install --upgrade pip setuptools

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt 

# Copy the rest of the app files into the container
COPY . /app/

# Expose the port that your FastAPI app will run on
EXPOSE 8000

# Command to run the FastAPI app with Uvicorn when the container starts
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
