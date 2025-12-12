# Getting Started with the Credit Scoring Platform

Welcome to the Alternative Credit Scoring Platform! This guide will help you set up your local development environment and get started with the project.

## Prerequisites

Before getting started, ensure you have the following installed on your system:

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Python 3.13.x](https://www.python.org/downloads/) (specifically 3.13.x due to Apache Airflow compatibility)
- [Poetry](https://python-poetry.org/docs/#installation) (optional, but recommended)

## Quick Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd credit-scoring-platform
```

### 2. Environment Setup

1. Copy the environment template to a new `.env` file:
   ```bash
   cp .env.template .env
   ```
   
2. Review and customize the environment variables in the `.env` file if needed.

### 3. Start Services with Docker Compose

Run the following command to start all required services:

```bash
docker-compose up -d
```

This will start:
- PostgreSQL database (for MLflow tracking)
- MinIO (for artifact storage)
- MLflow tracking server

> **Note**: The services will be available at:
> - PostgreSQL: `localhost:5432`
> - MinIO: `http://localhost:9000` (console at `http://localhost:9001`)
> - MLflow: `http://localhost:5000`

### 4. Install Python Dependencies

Install project dependencies using pip:

```bash
pip install -r requirements.txt
```

Or if you're using Poetry:

```bash
poetry install
```

## Project Structure

```
├── configs/          # Configuration files
├── data/
│   ├── processed/    # Processed datasets
│   ├── raw/          # Raw datasets
│   └── reference/    # Reference datasets
├── docs/             # Documentation files
├── notebooks/        # Jupyter notebooks for experimentation
├── scripts/          # Scripts for data processing, model training, etc.
├── src/              # Source code
│   ├── data/         # Data processing modules
│   ├── features/     # Feature engineering modules
│   ├── models/       # Model training and evaluation modules
│   ├── api/          # API implementation
│   └── monitoring/   # Monitoring and drift detection modules
├── tests/            # Unit and integration tests
├── docker-compose.yml # Docker services configuration
├── requirements.txt   # Python dependencies
├── .env.template     # Environment variables template
└── README.md         # Project overview
```

## Running the Application

### Data Processing

To process the raw data and create features:

```bash
python scripts/make_dataset.py
```

### Model Training

To train a model with MLflow tracking:

```bash
python src/models/train.py
```

### Starting the API

To start the prediction API:

```bash
cd src/api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Development Workflow

### 1. Setup your branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make changes

- Add/modify code in the appropriate directories
- Write/update tests in the `tests/` directory
- Update documentation as needed

### 3. Run tests

```bash
pytest
```

### 4. Format code

```bash
black .
isort .
```

### 5. Commit and push

```bash
git add .
git commit -m "Description of changes"
git push origin feature/your-feature-name
```

## Useful Commands

- **Start all services**: `docker-compose up -d`
- **Stop all services**: `docker-compose down`
- **View service logs**: `docker-compose logs -f`
- **Install dependencies**: `pip install -r requirements.txt`
- **Run tests**: `pytest`
- **Format code**: `black . && isort .`
- **Run linting**: `flake8 .`

## Troubleshooting

### Docker containers won't start

- Ensure Docker and Docker Compose are properly installed and running
- Check that required ports (5432, 9000, 5000) are not already in use
- Run `docker-compose logs` to see detailed error messages

### Dependency installation fails

- Ensure you're using Python 3.13.x (due to Apache Airflow compatibility)
- Consider using a virtual environment
- Check that your internet connection is stable

### MLflow tracking not working

- Verify that the PostgreSQL and MinIO services are running
- Confirm that the MLflow server is accessible at `http://localhost:5000`
- Check that environment variables are properly set in your `.env` file

## Next Steps

1. Explore the Jupyter notebooks in the `notebooks/` directory for initial data analysis
2. Review the configuration files in `configs/` to understand project settings
3. Check out the documentation in `docs/` for detailed information about each component
4. Run the example training script to see the end-to-end pipeline in action