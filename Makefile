# Project entry points. Every target runs through .tools/uv, which is a single
# binary checked into .tools/ rather than installed on the host.
UV := .tools/uv

.PHONY: install infra-up infra-down test lint agent gateway web dev clean

install:                ## Resolve dependencies and install the package editable
	$(UV) sync --extra dev
	$(UV) pip install -e . --quiet

infra-up:               ## Start LiveKit, Postgres and Redis
	docker compose -f infra/docker-compose.yml up -d
	@echo "LiveKit ws://localhost:7880  Postgres 5432  Redis 6379"

infra-down:
	docker compose -f infra/docker-compose.yml down

test:
	$(UV) run pytest -q

lint:
	$(UV) run ruff check src tests

gateway:                ## Session issuance and consent gate
	$(UV) run uvicorn avatar.gateway.app:app --reload --port 8000

agent:                  ## Realtime pipeline; joins a room as a participant
	$(UV) run python -m avatar.realtime.agent

web:
	cd apps/web && npm run dev

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
