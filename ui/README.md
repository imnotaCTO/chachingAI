# UI Prototype

This UI is wired to the local API server in `scripts/api_server.py`.

## Run locally
From the repo root, start the API server (requires `ODDS_API_KEY` and `BALLDONTLIE_API_KEY` in `.env`):

```powershell
python scripts/api_server.py --port 8000
```

In a second terminal, start the static UI server:

```powershell
cd ui
python -m http.server 5173
```

Then open `http://localhost:5173` in a browser.

## What it shows
- Game selection (left rail)
- Props list with add-to-parlay
- Parlay summary with model vs implied probabilities

## Config
- To point the UI at a different API server, edit `ui/config.js`.

## Troubleshooting
- If props fail to load, verify API keys in `.env`.
- The API server limits to 30 players per event by default; use `--max-players` to expand.
