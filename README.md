# Vessel

Personal task and calendar assistant with a PWA front-end and an LLM-powered chat interface.

## Infrastructure

- **Hosting**: [Fly.io](https://fly.io) — app `vessel-ravi`, region `sjc`
- **Database**: [Supabase](https://supabase.com) — PostgreSQL, accessed via `asyncpg`

## Stack

- **Backend**: Python / FastAPI
- **Frontend**: Vanilla JS PWA (service worker, offline-capable)
- **LLM**: Groq (`gpt-oss-120b`) for chat and skip-reason inference
- **Observability**: Arize Phoenix (LLM call tracing)

## Running locally

```bash
cp .env.example .env  # fill in DATABASE_URL, GROQ_API_KEY, etc.
pip install -e ".[dev]"
uvicorn vessel.main:app --reload
```

## Tests

```bash
# Unit + hermetic UI tests
pytest tests/ -k "not live"

# Live tests against the deployed app
pytest tests/test_pwa_ui_live.py
```

## Deploy

```bash
fly deploy
bash scripts/post_deploy_test.sh
```
