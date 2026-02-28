# ARKE Physical AI API

FastAPI backend for ARKE Physical AI that wraps the Arke API.

## Features

- FastAPI REST API
- CORS enabled
- Automatic token management
- Proxy to Arke API endpoints
- Health check endpoint

## Setup

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Docker

Build and run with Docker:
```bash
docker build -t arke-api .
docker run -p 8000:8000 arke-api
```

## API Endpoints

- `GET /` - Root endpoint with API info
- `GET /health` - Health check
- `GET /api/arke/{endpoint}` - Proxy GET requests to Arke API
- `POST /api/arke/refresh-token` - Manually refresh authentication token

## Documentation

Interactive API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
