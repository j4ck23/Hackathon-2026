# Use an official lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy and install dependencies first (Docker caches this layer if requirements.txt hasn't changed)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the app
COPY . .

# Tell Flask to be visible outside the container (0.0.0.0 instead of localhost)
ENV FLASK_RUN_HOST=0.0.0.0

# Run the app
CMD ["python", "app.py"]
