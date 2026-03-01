# ARKE Physical AI

A full-stack application for ARKE Physical AI, consisting of a FastAPI backend and React frontend.

## Architecture

- **API** (`/api`) - FastAPI backend that proxies requests to the Arke API
- **Web** (`/web`) - React frontend built with Vite

## Quick Start with Docker

The easiest way to run the entire application is using Docker Compose:

```bash
docker-compose up --build
```

This will start both services:
- **Web Frontend**: http://localhost:3000
- **API Backend**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### Troubleshooting Docker

If you get a "port already allocated" error:

1. Stop any existing containers:
   ```bash
   docker-compose down
   ```

2. Check if ports 3000 or 8000 are in use:
   ```bash
   lsof -i :8000
   lsof -i :3000
   ```

3. Kill the process or change the port mapping in `docker-compose.yml`

## Local Development

### API (FastAPI)

See [api/README.md](api/README.md) for detailed instructions.

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Web (React)

See [web/README.md](web/README.md) for detailed instructions.

```bash
cd web
npm install
npm run dev
```

The web app runs on http://localhost:3000 and proxies API requests to http://localhost:8000.

## Operator Alert Startup Sequence (Telegram)

When you want the QC fail -> Telegram operator flow to work end-to-end, start services in this order:

1. Start API backend:
   ```bash
   cd api
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. Start web frontend:
   ```bash
   cd web
   npm run dev
   ```

3. Start public tunnel to backend:
   ```bash
   cloudflared tunnel --url http://localhost:8000
   ```
   Copy the generated `https://...trycloudflare.com` URL.

3.a 
   set -a
   source .env
   set +a


4. Configure Telegram webhook (replace with your tunnel URL):
   ```bash
   curl -s -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/setWebhook" \
     -d "url=$TELEGRAM_PUBLIC_URL/api/telegram/webhook" \
     -d "secret_token=$TELEGRAM_WEBHOOK_SECRET"
   ```

5. Verify webhook health:
   ```bash
   curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getWebhookInfo"
   ```
   Expected: no `last_error_message`, and `url` matches your current tunnel.

Required `.env` keys:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_OPERATOR_CHAT_ID`
- `TELEGRAM_WEBHOOK_SECRET`

Do you need to set webhook every restart?
- If using `trycloudflare.com` quick tunnel: **Yes**, URL changes most runs.
- If using a stable domain/named tunnel: **No**, only reset if URL or secret changes.

## Project Structure

```
.
├── api/                    # FastAPI backend
│   ├── main.py            # FastAPI application
│   ├── request.py         # Arke API client
│   ├── requirements.txt   # Python dependencies
│   └── Dockerfile         # API container config
├── web/                   # React frontend
│   ├── src/              # React source code
│   ├── package.json      # Node dependencies
│   ├── vite.config.js    # Vite configuration
│   └── Dockerfile        # Web container config (Nginx)
└── docker-compose.yml    # Multi-container orchestration
```

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /api/arke/{endpoint}` - Proxy to Arke API
- `POST /api/arke/refresh-token` - Refresh auth token
- `GET /api/line/state` - Core line progression state for dashboard
- `POST /api/line/parts` - Create a tracked part
- `POST /api/line/parts/{part_id}/detection` - Push station updates from vision/manual tools

## Tech Stack

**Backend:**
- FastAPI
- Uvicorn
- Python 3.11

**Frontend:**
- React 18
- Vite
- Axios

**DevOps:**
- Docker
- Docker Compose
- Nginx (production)
