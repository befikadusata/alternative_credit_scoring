# Getting Started: Local Development Setup

This guide provides comprehensive instructions for setting up and running the Alternative Credit Scoring Platform locally on your machine using Docker Compose. This allows you to quickly get a full MLOps environment, including the FastAPI, MLflow, PostgreSQL, MinIO, and Redis services, up and running with a single command.

---

## Prerequisites

Before you begin, ensure you have the following installed on your system:

1.  **Git:** For cloning the repository.
    *   [Install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
2.  **Docker & Docker Compose:** For containerizing and orchestrating the services.
    *   [Install Docker Desktop](https://www.docker.com/products/docker-desktop/) (includes Docker Compose)

---

## 1. Setup and Installation

Follow these steps to get the project running locally:

1.  **Clone the Repository:**
    Open your terminal or command prompt and clone the project:
    ```bash
    git clone https://github.com/your-username/credit-scoring-platform.git
    cd credit-scoring-platform
    ```
    *(Replace `https://github.com/your-username/credit-scoring-platform.git` with the actual repository URL)*

2.  **Copy Environment File:**
    The project uses environment variables for configuration. Copy the template file:
    ```bash
    cp .env.template .env
    ```
    You can inspect and modify the `.env` file if you need to change any default settings (e.g., database passwords, port numbers).

3.  **Build and Run Services with Docker Compose:**
    Navigate to the root of the project directory and execute the following command. This will build the necessary Docker images and start all the services.
    ```bash
    docker-compose up --build -d
    ```
    *   `--build`: Rebuilds service images (useful if you've made changes to the `Dockerfile` or project dependencies).
    *   `-d`: Runs the services in detached mode (in the background).

    Wait a few moments for all services to start and become healthy. You can check their status with `docker-compose ps`.

---

## 2. Accessing Services

Once all services are up and running, you can access them via your web browser or API clients:

-   **MLflow UI:**
    *   **URL:** `http://localhost:5000`
    *   Here you can view experiment runs, logged metrics, parameters, and manage models in the MLflow Model Registry.

-   **Credit Scoring API (FastAPI):**
    *   **Base URL:** `http://localhost:8000`
    *   **Swagger UI (Interactive API Docs):** `http://localhost:8000/docs`
    *   **ReDoc (Alternative API Docs):** `http://localhost:8000/redoc`
    *   These interactive documentation pages allow you to explore the API endpoints and even make test requests directly from your browser.

-   **MinIO Console (Object Storage):**
    *   **URL:** `http://localhost:9001`
    *   **Username:** `minioadmin` (from `.env.template`)
    *   **Password:** `minioadmin123` (from `.env.template`)
    *   You can browse the buckets and objects stored by MLflow (e.g., model artifacts).

---

## 3. Making a Sample Prediction

You can test the API by sending a sample prediction request. First, ensure you have a model registered in MLflow (you'll need to run a training job first, see `src/models/train.py`).

Open a new terminal and use `curl` to make a prediction:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d 
{
           "loan_amnt": 10000,
           "int_rate": 12.5,
           "installment": 333.33,
           "emp_length": 7,
           "annual_inc": 60000,
           "dti": 20.0,
           "delinq_2yrs": 0,
           "inq_last_6mths": 1,
           "open_acc": 10,
           "pub_rec": 0,
           "revol_bal": 5000,
           "revol_util": 35.0,
           "total_acc": 25,
           "term": " 36 months",
           "grade": "B",
           "sub_grade": "B2",
           "home_ownership": "MORTGAGE",
           "verification_status": "Verified",
           "purpose": "debt_consolidation",
           "initial_list_status": "f"
         }
```
*(Note: The actual feature names and values might vary based on your model's training data. This is a generic example.)*

---

## 4. Stopping and Cleaning Up

To stop the running services:

```bash
docker-compose down
```
To stop services and remove all volumes (including database data and MinIO artifacts):

```bash
docker-compose down --volumes
```

---

