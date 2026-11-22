.PHONY: setup run run-gtzan serve similar check-data load try-scaler try-image-crop try-noise smoke data clean

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

# With the services running and songs stored (make run): the songs most like a file.
#   make similar FILE=mmfeaturestore/output/synthetic/genres_original/synthetic_bass/synthetic_bass.00000.wav
#   make similar FILE=path/to/image.png
similar:
	@test -n "$(FILE)" || { echo "Usage: make similar FILE=path/to/clip.wav (or an image)"; exit 2; }
	uv run --frozen python scripts/similar.py "$(FILE)"

# Your own music, in the GTZAN layout (see README, "Use your own data").
# check-data checks every file and stores nothing; load stores the features and searches them.
#   make check-data DIR=my_music
#   make load DIR=my_music
check-data:
	@test -n "$(DIR)" || { echo "Usage: make check-data DIR=path/to/folder"; exit 2; }
	uv run --frozen python scripts/own_data.py --dir "$(DIR)" --check-only

load: setup
	@test -n "$(DIR)" || { echo "Usage: make load DIR=path/to/folder"; exit 2; }
	uv run --frozen python scripts/own_data.py --dir "$(DIR)" --check-only
	$(COMPOSE) up -d --wait
	uv run --frozen python scripts/own_data.py --dir "$(DIR)"

# Try this: three measured experiments on synthetic clips (see README, "Try this"). No Docker.
try-scaler: setup
	uv run --frozen python experiments/scaler.py

try-image-crop: setup
	uv run --frozen python experiments/image_crop.py

try-noise: setup
	uv run --frozen python experiments/noise.py

# Encoder checks on synthetic clips. No Docker. The own-data tests: uv run pytest tests/test_own_data.py
smoke:
	uv sync --frozen
	uv run --frozen pytest -q tests/test_smoke.py

# GTZAN from Kaggle into mmfeaturestore/sample_data/ (1.2 GB). Needs a Kaggle API token in
# ~/.kaggle/kaggle.json, or KAGGLE_USERNAME and KAGGLE_KEY.
data: setup
	cd mmfeaturestore && uv run --frozen python download_data.py

# Stop the services and delete their volumes, and remove generated files (mmfeaturestore/output/,
# including the experiments' clips).
# Downloaded GTZAN data in mmfeaturestore/sample_data/ is kept.
clean:
	-$(COMPOSE) down -v --remove-orphans
	rm -rf mmfeaturestore/output .pytest_cache
