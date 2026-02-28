from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
import logging
from datetime import datetime, timezone
from uuid import uuid4
import json
import os
import threading
import time

from request import ArkeAPI
from scheduler import ProductionScheduler, SchedulingPolicy
from production import ProductionOrderManager
from database import init_db, get_db, ProductionState, create_production_state, get_production_state, finish_operation, get_all_production_states, get_active_order_on_line, clear_production_line, OPERATION_ORDER

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None
    np = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ARKE Physical AI API",
    description="FastAPI wrapper for ARKE Physical AI",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_db()
    logger.info("Database initialized")

# Initialize Arke API client, scheduler, and production manager
arke_client = ArkeAPI()
scheduler = ProductionScheduler(arke_client, policy=SchedulingPolicy.EDF)
production_manager = ProductionOrderManager(arke_client, scheduler)
STATIONS = ["SMT", "Reflow", "THT", "AOI", "Test", "Coating", "Pack"]
DETECTION_STATES = set(STATIONS + ["NOT_PRESENT"])
API_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(API_DIR)
DEFAULT_STATION_CONFIG_PATH = os.path.join(REPO_ROOT, "demo_data", "station_zones.json")
DEFAULT_TRACKER_PROFILE_PATH = os.path.join(REPO_ROOT, "demo_data", "line_tracker_profile.json")
DEFAULT_DEFECT_PROFILE_PATH = os.path.join(REPO_ROOT, "demo_data", "defect_camera_profile.json")
DEFAULT_QC_REFERENCE_IMAGE = os.path.join(
    REPO_ROOT, "demo_data", "pcb_images", "pcb_ref_image.png"
)


class PartState(BaseModel):
    part_id: str
    label: Optional[str] = None
    current_station: str = "NOT_PRESENT"
    current_station_since: Optional[str] = None
    progress_index: int = -1
    completed_stations: List[str] = Field(default_factory=list)
    history: List[Dict[str, Any]] = Field(default_factory=list)
    last_updated_at: str


class TransitionEvent(BaseModel):
    event_id: str
    part_id: str
    from_station: str
    to_station: str
    source: str = "manual"
    confidence: Optional[float] = None
    happened_at: str


class CreatePartRequest(BaseModel):
    part_id: Optional[str] = None
    label: Optional[str] = None


class DetectionUpdateRequest(BaseModel):
    station: str
    source: str = "vision"
    confidence: Optional[float] = None


class VisionLiveSettingsUpdate(BaseModel):
    source: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    min_area: Optional[int] = None
    score_decay: Optional[float] = None
    switch_score: Optional[float] = None
    debounce_frames: Optional[int] = None
    lower_h: Optional[int] = None
    lower_s: Optional[int] = None
    lower_v: Optional[int] = None
    upper_h: Optional[int] = None
    upper_s: Optional[int] = None
    upper_v: Optional[int] = None


class VisionDefectSettingsUpdate(BaseModel):
    source: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    reference_image_path: Optional[str] = None


class ZonePolygonUpdate(BaseModel):
    station: str
    points_norm: List[List[float]]


class VisionZonesUpdate(BaseModel):
    polygons: List[ZonePolygonUpdate]


class VisionAutoCalibrateRequest(BaseModel):
    station: str = "SMT"
    samples: int = 8


parts_state: Dict[str, PartState] = {}
transition_events: List[TransitionEvent] = []


def _parse_capture_source(raw: str):
    cleaned = str(raw).strip()
    return int(cleaned) if cleaned.isdigit() else cleaned


class LineVisionRuntime:
    def __init__(self):
        self._lock = threading.RLock()
        self.source = "0"
        self.width = 960
        self.height = 540
        self.min_area = 800
        self.score_decay = 0.92
        self.switch_score = 0.55
        self.debounce_frames = 5
        self.hsv = {
            "lower_h": 20,
            "lower_s": 80,
            "lower_v": 80,
            "upper_h": 136,
            "upper_s": 255,
            "upper_v": 255,
        }
        self.polygons = []
        self.station_scores = {station: 0.0 for station in STATIONS}
        self.current_station = "NOT_PRESENT"
        self.last_non_not_present_station = None
        self.candidate_station = None
        self.candidate_count = 0
        self.instant_candidates: List[str] = []
        self.last_frame_at: Optional[str] = None
        self.last_detected_centroid = None
        self.cap = None
        self.cap_source = None
        self._load_station_config()
        self._load_tracker_profile()

    def _ensure_cv2(self):
        if cv2 is None or np is None:
            raise HTTPException(status_code=503, detail="OpenCV/Numpy not available.")

    def _load_station_config(self):
        if not os.path.exists(DEFAULT_STATION_CONFIG_PATH):
            self.polygons = []
            return
        with open(DEFAULT_STATION_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        camera_source = data.get("camera_source")
        if camera_source is not None:
            self.source = str(camera_source)
        polygons = data.get("polygons", [])
        normalized = []
        for item in polygons:
            station = item.get("station")
            points_norm = item.get("points_norm", [])
            if station in STATIONS and len(points_norm) >= 3:
                normalized.append({"station": station, "points_norm": points_norm})
        self.polygons = normalized

    def _load_tracker_profile(self):
        if not os.path.exists(DEFAULT_TRACKER_PROFILE_PATH):
            return
        with open(DEFAULT_TRACKER_PROFILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        hsv = data.get("hsv", {})
        tracker = data.get("tracker", {})
        source = data.get("source")
        if source is not None:
            self.source = str(source)
        self.hsv.update({k: int(v) for k, v in hsv.items() if k in self.hsv})
        self.min_area = int(tracker.get("min_area", self.min_area))
        self.score_decay = float(tracker.get("score_decay", self.score_decay))
        self.switch_score = float(tracker.get("switch_score", self.switch_score))
        self.debounce_frames = int(tracker.get("debounce_frames", self.debounce_frames))

    def _save_tracker_profile(self):
        payload = {
            "version": 1,
            "source": self.source,
            "hsv": self.hsv,
            "tracker": {
                "min_area": self.min_area,
                "score_decay": self.score_decay,
                "switch_score": self.switch_score,
                "debounce_frames": self.debounce_frames,
                "trail_len": 40,
            },
            "config_path": "../demo_data/station_zones.json",
        }
        os.makedirs(os.path.dirname(DEFAULT_TRACKER_PROFILE_PATH), exist_ok=True)
        with open(DEFAULT_TRACKER_PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _open_capture_locked(self):
        self._ensure_cv2()
        requested_source = _parse_capture_source(self.source)
        if self.cap is not None and self.cap_source == requested_source:
            return
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cap = cv2.VideoCapture(requested_source)
        if not cap.isOpened():
            raise HTTPException(status_code=503, detail=f"Could not open live camera source: {self.source}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap = cap
        self.cap_source = requested_source

    def _polygons_px(self):
        polygons_px = []
        for item in self.polygons:
            points = np.array(
                [(int(x * self.width), int(y * self.height)) for x, y in item["points_norm"]],
                dtype=np.int32,
            )
            polygons_px.append({"station": item["station"], "points": points})
        return polygons_px

    def _build_hsv_mask(self, hsv_frame, lower_h, upper_h, lower_s, lower_v):
        if lower_h <= upper_h:
            lower = np.array([lower_h, lower_s, lower_v], dtype=np.uint8)
            upper = np.array([upper_h, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv_frame, lower, upper)
        else:
            lower_a = np.array([0, lower_s, lower_v], dtype=np.uint8)
            upper_a = np.array([upper_h, 255, 255], dtype=np.uint8)
            lower_b = np.array([lower_h, lower_s, lower_v], dtype=np.uint8)
            upper_b = np.array([179, 255, 255], dtype=np.uint8)
            mask = cv2.bitwise_or(cv2.inRange(hsv_frame, lower_a, upper_a), cv2.inRange(hsv_frame, lower_b, upper_b))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def _detect(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self._build_hsv_mask(
            hsv_frame,
            self.hsv["lower_h"],
            self.hsv["upper_h"],
            self.hsv["lower_s"],
            self.hsv["lower_v"],
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < self.min_area:
            return None, area
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return None, area
        return (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])), area

    def _candidate_stations(self, centroid, polygons_px):
        if centroid is None:
            return []
        candidates = []
        for poly in polygons_px:
            if cv2.pointPolygonTest(poly["points"], centroid, False) >= 0:
                candidates.append(poly["station"])
        return candidates

    def _build_station_mask(self, station: str):
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        found = False
        for polygon in self.polygons:
            if polygon["station"] != station:
                continue
            found = True
            points = np.array(
                [(int(x * self.width), int(y * self.height)) for x, y in polygon["points_norm"]],
                dtype=np.int32,
            )
            cv2.fillPoly(mask, [points], 255)
        if not found:
            raise HTTPException(
                status_code=400,
                detail=f"No polygon configured for station '{station}'. Save zone first.",
            )
        return mask

    def _update_station(self, candidates):
        for station in self.station_scores:
            self.station_scores[station] *= self.score_decay
        if not candidates:
            self.current_station = "NOT_PRESENT"
            self.candidate_station = None
            self.candidate_count = 0
            return
        increment = 1.0 / len(candidates)
        for station in candidates:
            self.station_scores[station] += increment
        dominant_station = max(self.station_scores, key=lambda station: self.station_scores[station])
        dominant_score = self.station_scores[dominant_station]
        if dominant_score < self.switch_score:
            return
        if dominant_station != self.current_station:
            if self.candidate_station == dominant_station:
                self.candidate_count += 1
            else:
                self.candidate_station = dominant_station
                self.candidate_count = 1
            if self.candidate_count >= self.debounce_frames:
                self.current_station = dominant_station
                self.last_non_not_present_station = dominant_station
                self.candidate_station = None
                self.candidate_count = 0
        else:
            self.candidate_station = None
            self.candidate_count = 0

    def _annotate(self, frame, polygons_px, centroid):
        for poly in polygons_px:
            station = poly["station"]
            points = poly["points"].reshape((-1, 1, 2))
            color = (0, 255, 0) if station == self.current_station else (255, 200, 0)
            cv2.polylines(frame, [points], True, color, 2)
            anchor = poly["points"][0]
            cv2.putText(frame, station, (int(anchor[0]) + 6, int(anchor[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        if centroid is not None:
            cv2.circle(frame, centroid, 8, (0, 0, 255), -1)
        cv2.putText(frame, f"inferred={self.current_station}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    def snapshot(self):
        with self._lock:
            self._open_capture_locked()
            ok, frame = self.cap.read()
            if not ok:
                raise HTTPException(status_code=503, detail="Could not read frame from live camera.")
            frame = cv2.resize(frame, (self.width, self.height))
            polygons_px = self._polygons_px()
            centroid, _ = self._detect(frame)
            self.last_detected_centroid = centroid
            self.instant_candidates = self._candidate_stations(centroid, polygons_px)
            self._update_station(self.instant_candidates)
            self._annotate(frame, polygons_px, centroid)
            self.last_frame_at = utc_now_iso()
            ok_img, encoded = cv2.imencode(".jpg", frame)
            if not ok_img:
                raise HTTPException(status_code=500, detail="Could not encode live frame.")
            return encoded.tobytes()

    def status(self):
        with self._lock:
            anchor = self.last_non_not_present_station
            if anchor in STATIONS:
                anchor_idx = STATIONS.index(anchor)
                completed = STATIONS[: anchor_idx + 1]
                remaining = STATIONS[anchor_idx + 1 :]
            else:
                completed = []
                remaining = STATIONS.copy()
            dominant_station = max(self.station_scores, key=lambda station: self.station_scores[station])
            return {
                "current_station": self.current_station,
                "phase_sequence": STATIONS,
                "completed_phases": completed,
                "remaining_phases": remaining,
                "scoreboard": self.station_scores,
                "dominant_station": dominant_station,
                "dominant_score": float(self.station_scores[dominant_station]),
                "instant_candidates": self.instant_candidates,
                "last_detected_centroid": self.last_detected_centroid,
                "last_frame_at": self.last_frame_at,
            }

    def get_settings(self):
        with self._lock:
            return {
                "source": self.source,
                "width": self.width,
                "height": self.height,
                "min_area": self.min_area,
                "score_decay": self.score_decay,
                "switch_score": self.switch_score,
                "debounce_frames": self.debounce_frames,
                **self.hsv,
            }

    def update_settings(self, payload: VisionLiveSettingsUpdate):
        with self._lock:
            incoming = payload.model_dump(exclude_none=True)
            source_changed = False
            for key, value in incoming.items():
                if key == "source":
                    self.source = str(value)
                    source_changed = True
                elif key in self.hsv:
                    self.hsv[key] = int(value)
                elif key in {"width", "height", "min_area", "debounce_frames"}:
                    setattr(self, key, int(value))
                elif key in {"score_decay", "switch_score"}:
                    setattr(self, key, float(value))
            if self.cap is not None and (source_changed or "width" in incoming or "height" in incoming):
                self.cap.release()
                self.cap = None
                self.cap_source = None
            self._save_tracker_profile()
            return self.get_settings()

    def get_zones(self):
        with self._lock:
            return {
                "stations": STATIONS,
                "polygons": self.polygons,
                "camera_source": self.source,
                "canvas_size": {"width": self.width, "height": self.height},
                "config_path": DEFAULT_STATION_CONFIG_PATH,
            }

    def update_zones(self, payload: VisionZonesUpdate):
        with self._lock:
            normalized = []
            for polygon in payload.polygons:
                if polygon.station not in STATIONS or len(polygon.points_norm) < 3:
                    raise HTTPException(status_code=400, detail=f"Invalid zone payload for station {polygon.station}")
                clean_points = []
                for point in polygon.points_norm:
                    if len(point) != 2:
                        raise HTTPException(status_code=400, detail=f"Invalid point format: {point}")
                    x = float(point[0])
                    y = float(point[1])
                    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                        raise HTTPException(status_code=400, detail=f"Normalized points must be in [0,1]. Got: {point}")
                    clean_points.append([x, y])
                normalized.append({"station": polygon.station, "points_norm": clean_points})
            self.polygons = normalized
            payload_data = {
                "version": 1,
                "stations": STATIONS,
                "camera_source": self.source,
                "canvas_size": {"width": self.width, "height": self.height},
                "polygons": self.polygons,
            }
            os.makedirs(os.path.dirname(DEFAULT_STATION_CONFIG_PATH), exist_ok=True)
            with open(DEFAULT_STATION_CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(payload_data, f, indent=2)
            return self.get_zones()

    def auto_calibrate_hsv(self, station: str, samples: int = 8):
        with self._lock:
            self._ensure_cv2()
            if station not in STATIONS:
                raise HTTPException(status_code=400, detail=f"Invalid station: {station}")
            safe_samples = min(max(int(samples), 3), 20)
            self._open_capture_locked()
            station_mask = self._build_station_mask(station)
            frames = []
            for _ in range(safe_samples):
                ok, frame = self.cap.read()
                if ok:
                    frames.append(cv2.resize(frame, (self.width, self.height)))
                time.sleep(0.03)
            if not frames:
                raise HTTPException(status_code=503, detail="Could not read frames for calibration.")

            hsv_pixels = []
            for frame in frames:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                pix = hsv[station_mask > 0]
                if pix.size > 0:
                    hsv_pixels.append(pix)

            if not hsv_pixels:
                raise HTTPException(
                    status_code=400,
                    detail=f"Station mask for '{station}' produced no pixels.",
                )

            pix = np.concatenate(hsv_pixels, axis=0)
            # Keep saturated/bright samples to avoid background noise.
            valid = pix[(pix[:, 1] > 25) & (pix[:, 2] > 25)]
            if valid.size == 0:
                valid = pix

            h = valid[:, 0].astype(np.int32)
            s = valid[:, 1].astype(np.int32)
            v = valid[:, 2].astype(np.int32)
            hist = np.bincount(h, minlength=180)
            center = int(np.argmax(hist))

            # Dynamic hue half-width from spread around dominant hue.
            # Wrap-aware diff to handle red boundary.
            delta = np.abs(h - center)
            delta = np.minimum(delta, 180 - delta)
            half_width = int(np.clip(np.percentile(delta, 90), 10, 35))

            lower_h = int((center - half_width) % 180)
            upper_h = int((center + half_width) % 180)
            lower_s = int(np.clip(np.percentile(s, 25) - 10, 25, 170))
            lower_v = int(np.clip(np.percentile(v, 25) - 10, 25, 170))

            self.hsv["lower_h"] = lower_h
            self.hsv["upper_h"] = upper_h
            self.hsv["lower_s"] = lower_s
            self.hsv["lower_v"] = lower_v
            self.hsv["upper_s"] = 255
            self.hsv["upper_v"] = 255
            self._save_tracker_profile()
            return {
                "station": station,
                "samples_used": len(frames),
                "applied_hsv": self.hsv,
                "metrics": {
                    "dominant_h": center,
                    "hue_half_width": half_width,
                    "num_pixels": int(valid.shape[0]),
                },
            }


class DefectVisionRuntime:
    def __init__(self):
        self._lock = threading.RLock()
        self.source = "1"
        self.width = 960
        self.height = 540
        self.reference_image_path = DEFAULT_QC_REFERENCE_IMAGE
        self.cap = None
        self.cap_source = None
        self._load_profile()

    def _load_profile(self):
        if not os.path.exists(DEFAULT_DEFECT_PROFILE_PATH):
            return
        with open(DEFAULT_DEFECT_PROFILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("source") is not None:
            self.source = str(data["source"])
        if data.get("width") is not None:
            self.width = int(data["width"])
        if data.get("height") is not None:
            self.height = int(data["height"])
        if data.get("reference_image_path"):
            self.reference_image_path = str(data["reference_image_path"])

    def _save_profile(self):
        payload = {
            "version": 1,
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "reference_image_path": self.reference_image_path,
        }
        os.makedirs(os.path.dirname(DEFAULT_DEFECT_PROFILE_PATH), exist_ok=True)
        with open(DEFAULT_DEFECT_PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _ensure_cv2(self):
        if cv2 is None:
            raise HTTPException(status_code=503, detail="OpenCV not available.")

    def _open_capture_locked(self):
        self._ensure_cv2()
        requested_source = _parse_capture_source(self.source)
        if self.cap is not None and self.cap_source == requested_source:
            return
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cap = cv2.VideoCapture(requested_source)
        if not cap.isOpened():
            raise HTTPException(status_code=503, detail=f"Could not open defect camera source: {self.source}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap = cap
        self.cap_source = requested_source

    def frame(self):
        with self._lock:
            self._open_capture_locked()
            ok, frame = self.cap.read()
            if not ok:
                raise HTTPException(status_code=503, detail="Could not read frame from defect camera.")
            frame = cv2.resize(frame, (self.width, self.height))
            cv2.putText(frame, "Defect Camera Live", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            ok_img, encoded = cv2.imencode(".jpg", frame)
            if not ok_img:
                raise HTTPException(status_code=500, detail="Could not encode defect frame.")
            return encoded.tobytes()

    def status(self):
        with self._lock:
            return {
                "source": self.source,
                "width": self.width,
                "height": self.height,
                "reference_image_path": self.reference_image_path,
                "reference_exists": os.path.exists(self.reference_image_path),
            }

    def update_settings(self, payload: VisionDefectSettingsUpdate):
        with self._lock:
            incoming = payload.model_dump(exclude_none=True)
            if "source" in incoming:
                self.source = str(incoming["source"])
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                    self.cap_source = None
            if "width" in incoming:
                self.width = int(incoming["width"])
            if "height" in incoming:
                self.height = int(incoming["height"])
            if "reference_image_path" in incoming:
                path = incoming["reference_image_path"]
                if not os.path.isabs(path):
                    path = os.path.join(REPO_ROOT, path)
                self.reference_image_path = path
            self._save_profile()
            return self.status()


line_vision_runtime = LineVisionRuntime()
defect_vision_runtime = DefectVisionRuntime()


class ArkeRequest(BaseModel):
    endpoint: str
    method: Optional[str] = "GET"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_part_exists(part_id: str) -> PartState:
    part = parts_state.get(part_id)
    if part is None:
        raise HTTPException(status_code=404, detail=f"Part not found: {part_id}")
    return part


def update_part_station(
    part: PartState,
    to_station: str,
    source: str,
    confidence: Optional[float],
) -> Optional[TransitionEvent]:
    if to_station not in DETECTION_STATES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid station '{to_station}'. Allowed: {sorted(DETECTION_STATES)}",
        )

    from_station = part.current_station
    if from_station == to_station:
        part.last_updated_at = utc_now_iso()
        return None

    now_iso = utc_now_iso()

    if part.history and "exited_at" not in part.history[-1]:
        part.history[-1]["exited_at"] = now_iso

    if to_station != "NOT_PRESENT":
        part.history.append({"station": to_station, "entered_at": now_iso})

    part.current_station = to_station
    part.current_station_since = None if to_station == "NOT_PRESENT" else now_iso
    part.last_updated_at = now_iso

    if to_station in STATIONS:
        station_idx = STATIONS.index(to_station)
        if station_idx > part.progress_index:
            part.progress_index = station_idx
            part.completed_stations = STATIONS[: station_idx + 1]

    event = TransitionEvent(
        event_id=f"evt_{uuid4().hex[:12]}",
        part_id=part.part_id,
        from_station=from_station,
        to_station=to_station,
        source=source,
        confidence=confidence,
        happened_at=now_iso,
    )
    transition_events.append(event)
    return event


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ARKE Physical AI API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/api/arke/{endpoint:path}")
async def proxy_arke_get(endpoint: str):
    """
    Proxy GET requests to Arke API
    
    Args:
        endpoint: The API endpoint path (without base URL)
    
    Returns:
        JSON response from Arke API
    """
    try:
        # Ensure endpoint starts with /
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        
        logger.info(f"Proxying GET request to: {endpoint}")
        result = arke_client.get(endpoint)
        return result
    except Exception as e:
        logger.error(f"Error proxying request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/arke/refresh-token")
async def refresh_token():
    """
    Manually refresh the Arke API token
    
    Returns:
        Status message
    """
    try:
        arke_client.login()
        return {"message": "Token refreshed successfully"}
    except Exception as e:
        logger.error(f"Error refreshing token: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scheduler/orders")
async def get_sales_orders():
    """
    Get all accepted sales orders
    
    Returns:
        List of sales orders with scheduling info
    """
    try:
        logger.info("Fetching active sales orders")
        orders = arke_client.get("/sales/order/_active")
        
        sales_orders = scheduler.parse_sales_orders(orders)
        
        return {
            "count": len(sales_orders),
            "orders": [
                {
                    "id": o.id,
                    "order_number": o.order_number,
                    "customer": o.customer,
                    "product": o.product_name,
                    "quantity": o.quantity,
                    "deadline": o.deadline.strftime("%Y-%m-%d"),
                    "priority": o.priority,
                    "production_days": scheduler.calculate_production_time(o.product_name, o.quantity)[0]
                }
                for o in sales_orders
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching sales orders: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scheduler/schedule")
async def get_production_schedule():
    """
    Generate production schedule using EDF policy
    
    Returns:
        Production schedule with conflict detection
    """
    try:
        logger.info("Generating production schedule")
        
        orders = arke_client.get("/sales/order/_active")
        sales_orders = scheduler.parse_sales_orders(orders)
        
        # Create EDF schedule
        production_plans = scheduler.create_edf_schedule(sales_orders)
        
        # Detect conflicts
        conflicts = scheduler.detect_conflicts(sales_orders)
        
        # Generate summary
        summary = scheduler.generate_schedule_summary(production_plans, conflicts)
        
        return {
            "policy": "EDF (Earliest Deadline First)",
            "generated_at": scheduler.CURRENT_DATE.isoformat(),
            "production_plans": [
                {
                    "order_number": p.sales_order_number,
                    "customer": p.customer,
                    "product": p.product_name,
                    "quantity": p.quantity,
                    "priority": p.priority,
                    "starts_at": p.starts_at.isoformat(),
                    "ends_at": p.ends_at.isoformat(),
                    "deadline": p.deadline.isoformat(),
                    "reasoning": p.reasoning
                }
                for p in production_plans
            ],
            "conflicts": conflicts,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error generating schedule: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scheduler/conflicts")
async def detect_scheduling_conflicts():
    """
    Detect scheduling conflicts (SO-005 vs SO-003)
    
    Returns:
        List of detected conflicts
    """
    try:
        logger.info("Detecting scheduling conflicts")
        
        orders = arke_client.get("/sales/order/_active")
        sales_orders = scheduler.parse_sales_orders(orders)
        conflicts = scheduler.detect_conflicts(sales_orders)
        
        return {
            "conflicts_found": len(conflicts),
            "conflicts": conflicts
        }
    except Exception as e:
        logger.error(f"Error detecting conflicts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/production/create")
async def create_production_orders():
    """
    Create production orders in Arke from EDF schedule
    
    Returns:
        List of created production orders
    """
    try:
        logger.info("Creating production orders")
        
        orders = arke_client.get("/sales/order/_active")
        sales_orders = scheduler.parse_sales_orders(orders)
        production_plans = scheduler.create_edf_schedule(sales_orders)
        
        # Create production orders in Arke
        results = production_manager.create_full_schedule(production_plans)
        
        return {
            "created": len(results),
            "production_orders": results
        }
    except Exception as e:
        logger.error(f"Error creating production orders: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/production/{production_order_id}/confirm")
async def confirm_production_order(production_order_id: str):
    """
    Confirm a production order after approval
    Moves to in_progress and unlocks first phase
    
    Args:
        production_order_id: ID of the production order to confirm
        
    Returns:
        Confirmed production order
    """
    try:
        logger.info(f"Confirming production order {production_order_id}")
        result = production_manager.confirm_production_order(production_order_id)
        return result
    except Exception as e:
        logger.error(f"Error confirming production order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/production/phase/{phase_id}/start")
async def start_phase(phase_id: str):
    """
    Start a production phase
    
    Args:
        phase_id: ID of the phase to start
        
    Returns:
        Updated phase data
    """
    try:
        logger.info(f"Starting phase {phase_id}")
        result = production_manager.start_phase(phase_id)
        return result
    except Exception as e:
        logger.error(f"Error starting phase: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/line/parts")
async def create_part(payload: CreatePartRequest):
    part_id = payload.part_id or f"PART-{uuid4().hex[:6].upper()}"
    if part_id in parts_state:
        raise HTTPException(status_code=400, detail=f"Part already exists: {part_id}")

    now_iso = utc_now_iso()
    part = PartState(
        part_id=part_id,
        label=payload.label,
        last_updated_at=now_iso,
    )
    parts_state[part_id] = part
    return {"part": part}


@app.post("/api/line/parts/{part_id}/detection")
async def update_detection(part_id: str, payload: DetectionUpdateRequest):
    part = ensure_part_exists(part_id)
    event = update_part_station(
        part=part,
        to_station=payload.station,
        source=payload.source,
        confidence=payload.confidence,
    )
    return {"part": part, "event": event}


@app.get("/api/line/state")
async def get_line_state():
    by_station: Dict[str, List[PartState]] = {station: [] for station in STATIONS}
    not_present: List[PartState] = []
    for part in parts_state.values():
        if part.current_station in STATIONS:
            by_station[part.current_station].append(part)
        else:
            not_present.append(part)

    for station in STATIONS:
        by_station[station].sort(key=lambda p: p.last_updated_at, reverse=True)
    not_present.sort(key=lambda p: p.last_updated_at, reverse=True)

    return {
        "stations": STATIONS,
        "parts": sorted(parts_state.values(), key=lambda p: p.last_updated_at, reverse=True),
        "by_station": by_station,
        "not_present": not_present,
        "recent_events": transition_events[-30:],
    }


@app.get("/api/line/events")
async def get_line_events(limit: int = 50):
    safe_limit = min(max(limit, 1), 500)
    return {"events": transition_events[-safe_limit:]}


@app.post("/api/line/reset")
async def reset_line_state():
    parts_state.clear()
    transition_events.clear()
    return {"message": "Line state reset."}


@app.get("/api/vision/live/settings")
async def get_live_vision_settings():
    return line_vision_runtime.get_settings()


@app.post("/api/vision/live/settings")
async def update_live_vision_settings(payload: VisionLiveSettingsUpdate):
    return line_vision_runtime.update_settings(payload)


@app.get("/api/vision/live/zones")
async def get_live_vision_zones():
    return line_vision_runtime.get_zones()


@app.post("/api/vision/live/zones")
async def update_live_vision_zones(payload: VisionZonesUpdate):
    return line_vision_runtime.update_zones(payload)


@app.get("/api/vision/live/status")
async def get_live_vision_status():
    return line_vision_runtime.status()


@app.post("/api/vision/live/auto-calibrate")
async def auto_calibrate_live_hsv(payload: VisionAutoCalibrateRequest):
    return line_vision_runtime.auto_calibrate_hsv(
        station=payload.station,
        samples=payload.samples,
    )


@app.get("/api/vision/live/frame")
async def get_live_vision_frame():
    image_bytes = line_vision_runtime.snapshot()
    return Response(content=image_bytes, media_type="image/jpeg")


@app.get("/api/vision/defect/status")
async def get_defect_vision_status():
    return defect_vision_runtime.status()


@app.post("/api/vision/defect/settings")
async def update_defect_vision_settings(payload: VisionDefectSettingsUpdate):
    return defect_vision_runtime.update_settings(payload)


@app.get("/api/vision/defect/frame")
async def get_defect_vision_frame():
    image_bytes = defect_vision_runtime.frame()
    return Response(content=image_bytes, media_type="image/jpeg")


@app.get("/api/vision/defect/reference-image")
async def get_defect_reference_image():
    status = defect_vision_runtime.status()
    path = status["reference_image_path"]
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Reference image not found: {path}")
    return FileResponse(path, media_type="image/png")


@app.post("/api/production/phase/{phase_id}/complete")
async def complete_phase(phase_id: str):
    """
    Complete a production phase
    
    Args:
        phase_id: ID of the phase to complete
        
    Returns:
        Updated phase data
    """
    try:
        logger.info(f"Completing phase {phase_id}")
        result = production_manager.complete_phase(phase_id)
        return result
    except Exception as e:
        logger.error(f"Error completing phase: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Production State Endpoints

@app.get("/api/production-state")
async def get_all_states(db: Session = Depends(get_db)):
    """
    Get all production states
    
    Returns:
        List of all production states
    """
    try:
        states = get_all_production_states(db)
        return {
            "count": len(states),
            "states": [
                {
                    "production_order": state.active_po,
                    "smt": {
                        "required": state.op_smt_required,
                        "finished_at": state.op_smt_finished_at.isoformat() if state.op_smt_finished_at else None
                    },
                    "reflow": {
                        "required": state.op_reflow_required,
                        "finished_at": state.op_reflow_finished_at.isoformat() if state.op_reflow_finished_at else None
                    },
                    "tht": {
                        "required": state.op_tht_required,
                        "finished_at": state.op_tht_finished_at.isoformat() if state.op_tht_finished_at else None
                    },
                    "aoi": {
                        "required": state.op_aoi_required,
                        "finished_at": state.op_aoi_finished_at.isoformat() if state.op_aoi_finished_at else None
                    },
                    "test": {
                        "required": state.op_test_required,
                        "finished_at": state.op_test_finished_at.isoformat() if state.op_test_finished_at else None
                    },
                    "coating": {
                        "required": state.op_coating_required,
                        "finished_at": state.op_coating_finished_at.isoformat() if state.op_coating_finished_at else None
                    },
                    "pack": {
                        "required": state.op_pack_required,
                        "finished_at": state.op_pack_finished_at.isoformat() if state.op_pack_finished_at else None
                    }
                }
                for state in states
            ]
        }
    except Exception as e:
        logger.error(f"Error getting production states: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# IMPORTANT: More specific routes MUST come before general :path routes
# Otherwise FastAPI will match the general route first and never reach the specific ones

@app.post("/api/production-state/{production_order_id:path}/operation/{operation}/start")
async def start_operation(production_order_id: str, operation: str, db: Session = Depends(get_db)):
    """No longer needed - operations are set as required when order is loaded"""
    return {"message": "Operations are now set as required at load time. Use /complete to mark finished."}


@app.post("/api/production-state/{production_order_id:path}/operation/{operation}/complete")
async def complete_operation(production_order_id: str, operation: str, db: Session = Depends(get_db)):
    """
    Finish an operation for the active order.
    Enforces sequential order and required-checks.
    """
    try:
        state = finish_operation(db, production_order_id, operation)
        return {"message": f"Operation {operation} completed for {production_order_id}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error completing operation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# General routes with :path must come AFTER specific routes to avoid conflicts

@app.post("/api/production-state/{production_order_id:path}")
async def create_state(production_order_id: str, db: Session = Depends(get_db)):
    """
    Load an order onto the production line.
    ENFORCES SINGLE ORDER: Only ONE record exists in the table at any time.
    Loading a new order overwrites the existing record.
    
    Args:
        production_order_id: Production order ID
        
    Returns:
        Created production state
    """
    try:
        # Get current order on line (if any)
        existing_state = db.query(ProductionState).first()
        replaced_order = existing_state.active_po if existing_state else None
        
        # Create new state (this will delete existing record and create new one)
        state = create_production_state(db, production_order_id)
        
        return {
            "message": f"Production state created for {production_order_id}",
            "production_order": state.active_po,
            "replaced": replaced_order
        }
    except Exception as e:
        logger.error(f"Error creating production state: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/production-state/{production_order_id:path}")
async def get_state(production_order_id: str, db: Session = Depends(get_db)):
    """
    Get production state for a specific order
    
    Args:
        production_order_id: Production order ID
        
    Returns:
        Production state
    """
    try:
        state = get_production_state(db, production_order_id)
        if not state:
            raise HTTPException(status_code=404, detail=f"Production state not found for {production_order_id}")
        
        return {
            "production_order": state.active_po,
            "smt": {
                "required": state.op_smt_required,
                "finished_at": state.op_smt_finished_at.isoformat() if state.op_smt_finished_at else None
            },
            "reflow": {
                "required": state.op_reflow_required,
                "finished_at": state.op_reflow_finished_at.isoformat() if state.op_reflow_finished_at else None
            },
            "tht": {
                "required": state.op_tht_required,
                "finished_at": state.op_tht_finished_at.isoformat() if state.op_tht_finished_at else None
            },
            "aoi": {
                "required": state.op_aoi_required,
                "finished_at": state.op_aoi_finished_at.isoformat() if state.op_aoi_finished_at else None
            },
            "test": {
                "required": state.op_test_required,
                "finished_at": state.op_test_finished_at.isoformat() if state.op_test_finished_at else None
            },
            "coating": {
                "required": state.op_coating_required,
                "finished_at": state.op_coating_finished_at.isoformat() if state.op_coating_finished_at else None
            },
            "pack": {
                "required": state.op_pack_required,
                "finished_at": state.op_pack_finished_at.isoformat() if state.op_pack_finished_at else None
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting production state: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
