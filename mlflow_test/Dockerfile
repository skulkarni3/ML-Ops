# Step 1: Start with a base image containing Python
FROM python:3.9-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Install MLflow and other dependencies
RUN pip install --no-cache-dir mlflow

# Step 4: Expose the port that MLflow will run on
EXPOSE 5000

# Step 5: Command to run MLflow server on startup
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
