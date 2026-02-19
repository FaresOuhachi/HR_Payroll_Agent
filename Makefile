# =============================================================================
# Makefile — Common commands for the HR Payroll Agent System
# =============================================================================
# CONCEPT: Makefiles provide short aliases for common commands.
# Instead of remembering long commands, you type: make run, make test, etc.
# =============================================================================

.PHONY: setup run test migrate seed docker-up docker-down clean

# --- Setup ---
# Install all Python dependencies
setup:
	pip install -r requirements.txt

# --- Docker ---
# Start PostgreSQL and Redis containers
docker-up:
	docker compose up -d
	@echo "Waiting for PostgreSQL to be ready..."
	@sleep 3
	@echo "Services are up! PostgreSQL on :5432, Redis on :6379"

# Stop and remove containers
docker-down:
	docker compose down

# --- Database ---
# Run Alembic migrations to create/update database tables
migrate:
	cd src && alembic -c ../alembic.ini upgrade head

# Generate a new migration from model changes
migration:
	cd src && alembic -c ../alembic.ini revision --autogenerate -m "$(msg)"

# --- Seed Data ---
# Populate database with sample employees and users
seed:
	python -m scripts.seed_data

# Ingest sample HR policy documents into vector store
ingest:
	python -m scripts.ingest_policies

# --- Run ---
# Start the FastAPI development server with auto-reload
run:
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# --- Test ---
# Run all tests
test:
	pytest tests/ -v

# Run tests with coverage report
test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

# --- Clean ---
# Remove Python cache files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
