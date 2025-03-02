# Use a lightweight Python base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements first for better caching
COPY api/requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire API source code
COPY api/ .

# Copy the trained model into the container
COPY model/linear_regression_model.pkl /model/linear_regression_model.pkl

# Expose the port FastAPI runs on
EXPOSE 8000

# Run the API with Gunicorn + Uvicorn for better performance
CMD ["python", "main.py"]
