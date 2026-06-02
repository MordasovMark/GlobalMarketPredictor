# AGENTS.md

## Cursor Cloud specific instructions

### Product overview

**Global Market Predictor** — single-repo app with:

| Layer | Path | Port |
|-------|------|------|
| React dashboard (primary UI) | `frontend/` | 5173 (Vite) |
| FastAPI API | `main.py` (mounts `api.py`) | 8000 |
| Streamlit dashboard (optional) | `dashboard.py` | 8501 |

There is no root `package.json`, Docker Compose, or Makefile. Standard commands live in `frontend/package.json` and `requirements.txt`.

### Running services (local E2E)

Use **tmux** for long-running dev servers. The VM has `python3` but not necessarily a `python` shim — always use `python3`.

**Backend** (from repo root):

```bash
python3 main.py
```

Serves OpenAPI docs at `http://127.0.0.1:8000/docs`. Core React routes: `/api/analyze`, `/api/portfolio/simulate`.

**Frontend** (from `frontend/`):

```bash
npm run dev -- --host 0.0.0.0
```

On localhost, the app targets `http://localhost:8000` for the API (see `resolveApiBaseUrl()` in `frontend/src/App.jsx`). Non-localhost builds use the hosted Render API.

### Lint / build

| Command | Where |
|---------|--------|
| `npm run lint` | `frontend/` (ESLint; repo may have pre-existing lint findings) |
| `npm run build` | `frontend/` |
| No Python test/lint runner configured | — |

### Optional / external dependencies

- **Finnhub**: browser calls Finnhub directly; set `VITE_FINNHUB_KEY` in `frontend/.env` if the baked-in fallback is unavailable.
- **yfinance**: backend needs outbound network for live OHLCV in `/api/analyze`.
- **Fear & Greed microservice** (`http://127.0.0.1:5000/api/fear-greed`): referenced in the UI but not implemented in this repo; gauge keeps a default value on failure.
- **`models/trained_ai_model.pkl`**: may fail to load at startup (`KeyError` in joblib); server still runs and `api.py` routes work. `/predict_ticker` in `main.py` may be degraded without a valid model.

### Streamlit (secondary UI)

```bash
streamlit run dashboard.py
```

Defaults to the remote Render API unless configured otherwise — not required for React E2E.
