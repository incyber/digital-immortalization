# Project entry points. Every target runs through .tools/uv, which is a single
# binary checked into .tools/ rather than installed on the host.
UV := .tools/uv

.PHONY: install infra-up infra-down test test-e2e latency seed lint agent gateway web clean

install:                ## Resolve dependencies and install the package editable
	$(UV) sync --extra dev
	$(UV) pip install -e . --quiet

infra-up:               ## Start LiveKit, Postgres, Redis and the Piper sidecar
	docker compose -f infra/docker-compose.yml up -d --build
	@echo "LiveKit ws://localhost:7880  Postgres 5432  Redis 6379  Piper 5050"

infra-down:
	docker compose -f infra/docker-compose.yml down

test:                   ## Unit and integration tests; no infrastructure needed
	$(UV) run pytest -q

test-e2e:               ## Real LiveKit and Ollama; requires make infra-up
	AGENT_LOG=$${AGENT_LOG:-/tmp/avatar-agent.log} E2E=1 $(UV) run pytest tests/e2e -q

latency:                ## Measure end of speech to first avatar audio
	E2E=1 $(UV) run pytest tests/e2e/test_latency.py -q -s

seed:                   ## Demo avatar with a verified consent record
	$(UV) run python scripts/seed.py

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
