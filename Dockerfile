# Use the official Python image
FROM python:3.11.0

# Set the working directory
WORKDIR /app

# Install system dependencies first
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*/tmp/*
#RUN pip install pandas
# Copy requirements first for better caching
COPY requirements.txt .


# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt

# Set DagsHub token as an environment variable
ENV MLFLOW_TRACKING_USERNAME="djamesdean"
ENV MLFLOW_TRACKING_PASSWORD="8e009ed06f4ac66d34599916803055c698bbb9bf"


# Copy the rest of the application
COPY . .

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]