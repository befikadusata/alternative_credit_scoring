# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Set environment variables for poetry
ENV POETRY_VERSION=1.7.1
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV PATH="$POETRY_HOME/bin:$PATH"

# Install poetry and build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    gcc \
    g++ \
    && curl -sSL https://install.python-poetry.org | python -

# Set the working directory in the container
WORKDIR /app

# Copy the dependency files
COPY poetry.lock pyproject.toml ./

# Install project dependencies
# --no-root is important to avoid installing the project itself as editable
# --no-dev is to avoid installing development dependencies
RUN poetry install --no-root --no-dev -vvv

# Copy the rest of the application code
COPY src/ /app/src/

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application
# Use 0.0.0.0 to make it accessible from outside the container
CMD ["poetry", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
