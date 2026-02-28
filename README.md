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
