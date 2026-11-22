.PHONY: setup run run-gtzan serve smoke data clean

# Uniform targets for The Playbooks. The README's commands (pip install -r requirements.txt,
# docker-compose up -d, python run.py) still work as printed.

COMPOSE = docker compose -f mmfeaturestore/docker-compose.yml

# Install the locked dependencies into .venv and download the ResNet-18 weights (45 MB).
setup:
	uv sync --frozen
	uv run --frozen python -c "import torchvision.models as m; m.resnet18(weights=m.ResNet18_Weights.IMAGENET1K_V1)"

# Start PostgreSQL and Milvus, store features for 12 SYNTHETIC clips (no GTZAN, no credentials),
# run a similarity search per genre and print the nearest songs.
run: setup
	$(COMPOSE) up -d --wait
	cd mmfeaturestore && uv run --frozen python quickstart.py

# The same on the real GTZAN data, after `make data`.
run-gtzan: setup
	$(COMPOSE) up -d --wait
	cd mmfeaturestore && uv run --frozen python quickstart.py --data sample_data --per-genre 5

# Start the services, then the API on http://localhost:8000.
serve: setup
	$(COMPOSE) up -d --wait
	cd mmfeaturestore && uv run --frozen python run.py

# Encoder checks on synthetic clips. No Docker.
smoke:
	uv sync --frozen
	uv run --frozen pytest -q tests

# GTZAN from Kaggle into mmfeaturestore/sample_data/ (1.2 GB). Needs a Kaggle API token in
# ~/.kaggle/kaggle.json, or KAGGLE_USERNAME and KAGGLE_KEY.
data: setup
	cd mmfeaturestore && uv run --frozen python download_data.py

# Stop the services and delete their volumes, and remove generated files.
# Downloaded GTZAN data in mmfeaturestore/sample_data/ is kept.
clean:
	-$(COMPOSE) down -v --remove-orphans
	rm -rf mmfeaturestore/output .pytest_cache
