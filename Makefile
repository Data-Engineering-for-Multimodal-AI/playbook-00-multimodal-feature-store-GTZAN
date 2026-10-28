.PHONY: smoke services data ingest evaluate api down
APP = cd mmfeaturestore &&

smoke:      ## tests (synthetic data; start the services first for the API test)
	$(APP) uv run pytest -q tests
services:   ## PostgreSQL and Milvus (Docker)
	$(APP) docker compose up -d --wait
data:       ## GTZAN from Kaggle (needs a Kaggle API token)
	$(APP) uv run python download_data.py
ingest: services
	$(APP) uv run python ingest_gtzan.py
evaluate:
	$(APP) uv run python evaluate.py --first-edition
api: services
	$(APP) uv run python run.py
down:
	$(APP) docker compose down
