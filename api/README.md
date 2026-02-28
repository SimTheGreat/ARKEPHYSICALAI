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
- `POST /api/line/parts` - Create/register a tracked part
- `POST /api/line/parts/{part_id}/detection` - Update part station detection
- `GET /api/line/state` - Snapshot of parts by station + recent events
- `GET /api/line/events` - Transition event history
- `POST /api/line/reset` - Reset line-state demo data

## Core Backend Module

`/api/line/*` is the shared backend core for line progression.
All modular features (line vision tracker, QC detection, fault modules, operator tools) can publish station updates through this contract, and the frontend dashboard reads from one unified state.

## Documentation

Interactive API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Quick Digital QC Test

Use `qc_compare.py` to compare a reference PCB image against a test image.

Install OpenCV locally if needed:
```bash
pip install opencv-contrib-python
```

PASS test (same image):
```bash
python qc_compare.py --reference ../pcb_reference.jpg --test ../pcb_reference.jpg --output ../qc_pass.png
```

FAIL test (altered image):
```bash
python qc_compare.py --reference ../pcb_reference.jpg --test ../pcb_altered.jpg --output ../qc_fail.png
```

The script prints JSON (`status`, changed ratio, defect boxes) and saves an annotated image.

For ArUco-stabilized comparison (recommended), keep corner IDs in this order:
`top-left,top-right,bottom-right,bottom-left`

Example:
```bash
python qc_compare.py \
  --reference ../demo_data/pcb_images/pcb_ref_image.png \
  --test ../demo_data/pcb_images/pcb_sample_ok.png \
  --aruco-dict DICT_5X5_100 \
  --aruco-ids 10,11,12,13 \
  --output ../demo_data/pcb_images/qc_result_ok.png
```

## Live Camera QC (Logitech)

Run real-time QC with webcam feed (defaults are preconfigured for your demo data):
```bash
python qc_live.py
```

Equivalent explicit command:
```bash
python qc_live.py \
  --reference ../demo_data/pcb_images/pcb_ref_image.png \
  --camera-index 0 \
  --aruco-dict DICT_5X5_100 \
  --aruco-ids 10,11,12,13 \
  --threshold 45 \
  --min-area 800 \
  --fail-ratio 0.003
```

If the Logitech camera is not index `0`, try `--camera-index 1` or `2`.

Controls:
- `q`: quit
- `s`: save current frame + JSON metrics to `../demo_data/pcb_images/live_outputs/`

## ArUco Debug Only

Use this to debug marker stability before any QC logic:
```bash
python aruco_debug_live.py
```

Optional:
```bash
python aruco_debug_live.py --camera-index 1 --aruco-ids 10,11,12,13 --aruco-dict DICT_5X5_100
```

## Live Line Station Tracker UI

Tracks a colored PCB proxy box and determines current station from saved polygons with temporal concentration scoring.

```bash
python line_tracker_ui.py --config ../demo_data/station_zones.json --source 0
```

If multiple polygons overlap due camera perspective, the tracker uses rolling station scores + debounce before switching station.
If object is removed from view, station state is set to `NOT_PRESENT`.
Use `Save Profile` / `Load Profile` in the UI to persist HSV and tracker thresholds.
