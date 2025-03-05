FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies first to leverage Docker layer cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first
COPY pyproject.toml poetry.lock ./

# Install Python dependencies without installing the package
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-root

# Copy the application code after installing dependencies to leverage Docker layer cache
COPY entity_matcher ./entity_matcher

# Install our package
RUN poetry install

# Make sure the model directory exists
RUN mkdir -p models

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "entity_matcher/api.py"]