from fastapi import FastAPI, HTTPException, Depends, Header
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
import requests

from request import ArkeAPI
from scheduler import ProductionScheduler, SchedulingPolicy
from production import ProductionOrderManager
from database import (
    init_db, get_db, ProductionState, create_production_state,
    get_production_state, finish_operation, get_all_production_states,
    get_active_order_on_line, clear_production_line, OPERATION_ORDER,
    save_schedule, get_persisted_schedule, get_production_logs,
    add_production_log, ProductionLog, ScheduleEntry,
    ArkePushRecord, get_push_status, get_push_record, upsert_push_record,
)
from database import init_db, get_db, ProductionState, create_production_state, get_production_state, finish_operation, get_all_production_states, get_active_order_on_line, clear_production_line, OPERATION_ORDER
from qc_compare import detect_aruco_points, parse_ids, build_board_mask_from_markers

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
DEFAULT_DEFECT_OUTPUT_DIR = os.path.join(REPO_ROOT, "demo_data", "pcb_images", "live_outputs")


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
    contrast_gain: Optional[float] = None
    saturation_gain: Optional[float] = None
    comparison_method: Optional[str] = None
    min_area: Optional[int] = None
    fail_ratio: Optional[float] = None


class ZonePolygonUpdate(BaseModel):
    station: str
    points_norm: List[List[float]]


class VisionZonesUpdate(BaseModel):
    polygons: List[ZonePolygonUpdate]


class VisionAutoCalibrateRequest(BaseModel):
    station: str = "SMT"
    samples: int = 8


class TelegramWebhookRequest(BaseModel):
    update_id: Optional[int] = None
    callback_query: Optional[Dict[str, Any]] = None
    message: Optional[Dict[str, Any]] = None


parts_state: Dict[str, PartState] = {}
transition_events: List[TransitionEvent] = []
qc_operator_decisions: Dict[str, Dict[str, Any]] = {}
latest_qc_operator_decision_id: Optional[str] = None


def _telegram_api_call(method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="TELEGRAM_BOT_TOKEN is not configured.")
    response = requests.post(
        f"https://api.telegram.org/bot{token}/{method}",
        json=payload,
        timeout=8,
    )
    response.raise_for_status()
    data = response.json()
    if not data.get("ok", False):
        raise RuntimeError(data.get("description", f"Telegram API {method} failed"))
    return data


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
        self.reference_image_path = os.path.relpath(DEFAULT_QC_REFERENCE_IMAGE, REPO_ROOT)
        self.aruco_dict = "DICT_5X5_100"
        self.aruco_ids = [10, 11, 12, 13]  # top-left, top-right, bottom-right, bottom-left
        self.lock_frames = 4
        self.threshold = 45
        self.min_area = 800
        self.fail_ratio = 0.003
        self.idle_reset_seconds = 2.5
        self.contrast_gain = 1.35
        self.saturation_gain = 1.25
        self.comparison_method = "hybrid"  # absdiff | edge_diff | hybrid
        self.cap = None
        self.cap_source = None
        self.last_frame_request_ts = 0.0
        self._reset_runtime_state()
        self._load_profile()

    def _reset_runtime_state(self):
        self.lock_count = 0
        self.locked = False
        self.qc_confirmed = False
        self.last_detected_ids: List[int] = []
        self.last_qc_result: Optional[Dict[str, Any]] = None
        self.locked_raw_frame: Optional[Any] = None
        self.locked_aligned_frame: Optional[Any] = None
        self.locked_annotated_frame: Optional[Any] = None
        self.locked_alignment_ok = False
        self.last_operator_notification: Optional[Dict[str, Any]] = None
        self.last_qc_artifacts: Optional[Dict[str, str]] = None

    def _normalize_reference_path(self, raw_path: str) -> str:
        path = str(raw_path).strip()
        if not path:
            return os.path.relpath(DEFAULT_QC_REFERENCE_IMAGE, REPO_ROOT)
        if os.path.isabs(path):
            # If this absolute path points into repo, persist as repo-relative path.
            try:
                rel = os.path.relpath(path, REPO_ROOT)
                if not rel.startswith(".."):
                    path = rel
            except Exception:
                pass
            return os.path.normpath(path if not os.path.isabs(path) else raw_path)
        return os.path.normpath(path)

    def _absolute_reference_path(self) -> str:
        if os.path.isabs(self.reference_image_path):
            return self.reference_image_path
        return os.path.normpath(os.path.join(REPO_ROOT, self.reference_image_path))

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
            self.reference_image_path = self._normalize_reference_path(str(data["reference_image_path"]))
        else:
            self.reference_image_path = os.path.relpath(DEFAULT_QC_REFERENCE_IMAGE, REPO_ROOT)
        if data.get("aruco_dict"):
            self.aruco_dict = str(data["aruco_dict"])
        if data.get("aruco_ids"):
            try:
                self.aruco_ids = parse_ids(str(data["aruco_ids"]))
            except Exception:
                pass
        if data.get("contrast_gain") is not None:
            self.contrast_gain = float(data["contrast_gain"])
        if data.get("saturation_gain") is not None:
            self.saturation_gain = float(data["saturation_gain"])
        if data.get("comparison_method"):
            self.comparison_method = str(data["comparison_method"])
        if data.get("min_area") is not None:
            self.min_area = int(data["min_area"])
        if data.get("fail_ratio") is not None:
            self.fail_ratio = float(data["fail_ratio"])

    def _save_profile(self):
        payload = {
            "version": 1,
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "reference_image_path": self._normalize_reference_path(self.reference_image_path),
            "aruco_dict": self.aruco_dict,
            "aruco_ids": ",".join(str(x) for x in self.aruco_ids),
            "contrast_gain": self.contrast_gain,
            "saturation_gain": self.saturation_gain,
            "comparison_method": self.comparison_method,
            "min_area": self.min_area,
            "fail_ratio": self.fail_ratio,
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

    def _load_reference(self):
        path = self._absolute_reference_path()
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Reference image not found: {path}")
        ref = cv2.imread(path, cv2.IMREAD_COLOR)
        if ref is None:
            raise HTTPException(status_code=500, detail=f"Could not read reference image: {path}")
        return ref

    def _align_to_reference(self, frame_bgr, ref_bgr):
        ref_markers = detect_aruco_points(ref_bgr, self.aruco_dict, self.aruco_ids)
        test_markers = detect_aruco_points(frame_bgr, self.aruco_dict, self.aruco_ids)
        common_ids = [marker_id for marker_id in self.aruco_ids if marker_id in ref_markers and marker_id in test_markers]

        if len(common_ids) < 1:
            return cv2.resize(frame_bgr, (ref_bgr.shape[1], ref_bgr.shape[0])), False, ref_markers, test_markers

        src_pts = np.concatenate([test_markers[marker_id] for marker_id in common_ids], axis=0)
        dst_pts = np.concatenate([ref_markers[marker_id] for marker_id in common_ids], axis=0)
        homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
        if homography is None:
            return cv2.resize(frame_bgr, (ref_bgr.shape[1], ref_bgr.shape[0])), False, ref_markers, test_markers
        aligned = cv2.warpPerspective(frame_bgr, homography, (ref_bgr.shape[1], ref_bgr.shape[0]))
        return aligned, True, ref_markers, test_markers

    def _evaluate_qc_on_aligned(self, aligned_bgr, aligned_ok: bool):
        ref_bgr = self._load_reference()
        ref_markers = detect_aruco_points(ref_bgr, self.aruco_dict, self.aruco_ids)

        ref_gray = self._preprocess_for_diff(ref_bgr)
        test_gray = self._preprocess_for_diff(aligned_bgr)
        ref_blur = cv2.GaussianBlur(ref_gray, (3, 3), 0)
        test_blur = cv2.GaussianBlur(test_gray, (3, 3), 0)

        kernel = np.ones((3, 3), np.uint8)

        diff = cv2.absdiff(ref_blur, test_blur)
        _, abs_mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        abs_mask = cv2.morphologyEx(abs_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        abs_mask = cv2.morphologyEx(abs_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        ref_edges = cv2.Canny(ref_blur, 45, 135)
        test_edges = cv2.Canny(test_blur, 45, 135)
        edge_mask = cv2.bitwise_xor(ref_edges, test_edges)
        edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        edge_mask = cv2.dilate(edge_mask, kernel, iterations=1)

        method = self.comparison_method
        if method == "edge_diff":
            mask = edge_mask
        elif method == "hybrid":
            # Keep only intensity changes that are also backed by edge change.
            mask = cv2.bitwise_and(abs_mask, edge_mask)
            # Fall back to abs-only mask if hybrid becomes too sparse.
            if np.count_nonzero(mask) < 100:
                mask = abs_mask
        else:
            mask = abs_mask

        board_mask = build_board_mask_from_markers(ref_bgr.shape, ref_markers, self.aruco_ids)
        if np.count_nonzero(board_mask) > 0:
            mask = cv2.bitwise_and(mask, board_mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        defect_regions = []
        annotated = aligned_bgr.copy()
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            defect_regions.append({"x": x, "y": y, "w": w, "h": h, "area": float(area)})
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)

        changed_pixels = int(np.count_nonzero(mask))
        total_pixels = int(mask.shape[0] * mask.shape[1])
        changed_ratio = changed_pixels / total_pixels if total_pixels else 0.0
        qc_status = "FAIL" if defect_regions or changed_ratio > self.fail_ratio else "PASS"

        if np.count_nonzero(board_mask) > 0:
            board_contours, _ = cv2.findContours(board_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, board_contours, -1, (0, 255, 255), 2)

        status_color = (0, 200, 0) if qc_status == "PASS" else (0, 0, 255)
        cv2.putText(annotated, f"QC: {qc_status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2, cv2.LINE_AA)
        cv2.putText(annotated, f"changed_ratio={changed_ratio:.5f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"aruco_aligned={aligned_ok} method={method} detected={self.last_detected_ids}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            "ink_normalized=on (white background removed)",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        result = {
            "status": qc_status,
            "changed_ratio": changed_ratio,
            "defect_count": len(defect_regions),
            "defect_regions": defect_regions,
            "aruco_aligned": aligned_ok,
            "aruco_detected": self.last_detected_ids,
            "contrast_gain": self.contrast_gain,
            "saturation_gain": self.saturation_gain,
            "comparison_method": method,
            "timestamp": utc_now_iso(),
        }
        return annotated, result

    def _preprocess_for_diff(self, image_bgr):
        # Increase saturation/contrast, then keep only dark "ink" regions as black
        # and force background to pure white to reduce lighting/shadow artifacts.
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.saturation_gain, 0, 255)
        boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        contrasted = cv2.convertScaleAbs(boosted, alpha=self.contrast_gain, beta=0)
        gray = cv2.cvtColor(contrasted, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu on inverted grayscale marks dark strokes/components as foreground.
        _, dark_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        normalized = np.full_like(gray, 255)
        normalized[dark_mask > 0] = 0
        return normalized

    def _save_qc_snapshot(self, raw_frame, annotated_frame, result):
        os.makedirs(DEFAULT_DEFECT_OUTPUT_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        raw_path = os.path.join(DEFAULT_DEFECT_OUTPUT_DIR, f"defect_lock_raw_{stamp}.png")
        annotated_path = os.path.join(DEFAULT_DEFECT_OUTPUT_DIR, f"defect_lock_result_{stamp}.png")
        json_path = os.path.join(DEFAULT_DEFECT_OUTPUT_DIR, f"defect_lock_result_{stamp}.json")
        cv2.imwrite(raw_path, raw_frame)
        cv2.imwrite(annotated_path, annotated_frame)
        payload = dict(result)
        payload["raw_image"] = raw_path
        payload["annotated_image"] = annotated_path
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.last_qc_artifacts = {
            "raw_image": raw_path,
            "annotated_image": annotated_path,
            "json_path": json_path,
        }

    def frame(self):
        with self._lock:
            now = time.time()
            if self.last_frame_request_ts and (now - self.last_frame_request_ts) > self.idle_reset_seconds:
                # Re-arm detector when feed has been inactive (e.g. outside Test phase).
                self._reset_runtime_state()
            self.last_frame_request_ts = now
            self._open_capture_locked()
            ok, frame = self.cap.read()
            if not ok:
                raise HTTPException(status_code=503, detail="Could not read frame from defect camera.")
            frame = cv2.resize(frame, (self.width, self.height))

            if not self.locked:
                detected = detect_aruco_points(frame, self.aruco_dict, self.aruco_ids)
                found_ids = [marker_id for marker_id in self.aruco_ids if marker_id in detected]
                self.last_detected_ids = sorted(found_ids)
                if len(found_ids) == len(self.aruco_ids):
                    self.lock_count += 1
                else:
                    self.lock_count = 0

                preview = frame.copy()
                if self.last_detected_ids:
                    for marker_id in self.last_detected_ids:
                        pts = detected[marker_id].astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(preview, [pts], True, (0, 255, 255), 2)
                        cx = int(np.mean(detected[marker_id][:, 0]))
                        cy = int(np.mean(detected[marker_id][:, 1]))
                        cv2.putText(preview, f"ID {marker_id}", (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

                status = "LOCKING" if self.lock_count > 0 else "SEARCHING"
                cv2.putText(preview, f"Defect Camera: {status}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(
                    preview,
                    f"found_ids={self.last_detected_ids} lock={self.lock_count}/{self.lock_frames}",
                    (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                if self.lock_count >= self.lock_frames:
                    ref_bgr = self._load_reference()
                    aligned, aligned_ok, _, _ = self._align_to_reference(frame, ref_bgr)
                    self.locked = True
                    self.qc_confirmed = True
                    self.locked_raw_frame = frame.copy()
                    self.locked_aligned_frame = aligned.copy()
                    self.locked_alignment_ok = bool(aligned_ok)
                    annotated, result = self._evaluate_qc_on_aligned(self.locked_aligned_frame, self.locked_alignment_ok)
                    self.last_qc_result = result
                    self.locked_annotated_frame = annotated.copy()
                    if self.locked_raw_frame is not None:
                        self._save_qc_snapshot(self.locked_raw_frame, self.locked_annotated_frame, result)
                    output_frame = self.locked_annotated_frame.copy()
                    cv2.putText(
                        output_frame,
                        "QC FROZEN - Re-arm by obstruction/idle",
                        (12, 142),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        (0, 255, 180),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    output_frame = preview
            else:
                output_frame = self.locked_annotated_frame.copy() if self.locked_annotated_frame is not None else frame.copy()
                cv2.putText(
                    output_frame,
                    "QC FROZEN - Re-arm by obstruction/idle",
                    (12, 142),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 255, 180),
                    2,
                    cv2.LINE_AA,
                )

            ok_img, encoded = cv2.imencode(".jpg", output_frame)
            if not ok_img:
                raise HTTPException(status_code=500, detail="Could not encode defect frame.")
            return encoded.tobytes()

    def status(self):
        with self._lock:
            absolute_reference_path = self._absolute_reference_path()
            current_qc_ts = self.last_qc_result.get("timestamp") if self.last_qc_result else None
            decision_for_current_qc = None
            if current_qc_ts:
                matches = [
                    item
                    for item in qc_operator_decisions.values()
                    if item.get("qc_result", {}).get("timestamp") == current_qc_ts
                ]
                if matches:
                    matches.sort(key=lambda item: item.get("created_at", ""), reverse=True)
                    decision_for_current_qc = matches[0]
            return {
                "source": self.source,
                "width": self.width,
                "height": self.height,
                "reference_image_path": self._normalize_reference_path(self.reference_image_path),
                "reference_exists": os.path.exists(absolute_reference_path),
                "aruco_detected_ids": self.last_detected_ids,
                "aruco_lock_count": self.lock_count,
                "aruco_lock_frames_required": self.lock_frames,
                "snapshot_locked": self.locked,
                "qc_confirmed": self.qc_confirmed,
                "qc_result": self.last_qc_result,
                "operator_decision": decision_for_current_qc,
                "last_operator_notification": self.last_operator_notification,
                "last_qc_artifacts": self.last_qc_artifacts,
                "contrast_gain": self.contrast_gain,
                "saturation_gain": self.saturation_gain,
                "comparison_method": self.comparison_method,
                "min_area": self.min_area,
                "fail_ratio": self.fail_ratio,
            }

    def update_settings(self, payload: VisionDefectSettingsUpdate):
        with self._lock:
            incoming = payload.model_dump(exclude_none=True)
            hard_reset = False
            if "source" in incoming:
                self.source = str(incoming["source"])
                hard_reset = True
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                    self.cap_source = None
            if "width" in incoming:
                self.width = int(incoming["width"])
                hard_reset = True
            if "height" in incoming:
                self.height = int(incoming["height"])
                hard_reset = True
            if "reference_image_path" in incoming:
                self.reference_image_path = self._normalize_reference_path(incoming["reference_image_path"])
                hard_reset = True
            if "contrast_gain" in incoming:
                self.contrast_gain = max(0.5, min(3.0, float(incoming["contrast_gain"])))
            if "saturation_gain" in incoming:
                self.saturation_gain = max(0.0, min(3.0, float(incoming["saturation_gain"])))
            if "comparison_method" in incoming:
                requested = str(incoming["comparison_method"]).strip().lower()
                if requested not in {"absdiff", "edge_diff", "hybrid"}:
                    raise HTTPException(status_code=400, detail="Invalid comparison_method. Use: absdiff, edge_diff, hybrid")
                self.comparison_method = requested
            if "min_area" in incoming:
                self.min_area = max(50, min(20000, int(incoming["min_area"])))
            if "fail_ratio" in incoming:
                self.fail_ratio = max(0.0001, min(0.2, float(incoming["fail_ratio"])))
            if hard_reset:
                self._reset_runtime_state()
            self._save_profile()
            return self.status()

    def notify_operator(self):
        global latest_qc_operator_decision_id
        with self._lock:
            if not self.last_qc_result or self.last_qc_result.get("status") != "FAIL":
                raise HTTPException(status_code=400, detail="Notify Operator is only available for QC FAIL.")

            decision_id = f"qcd_{uuid4().hex[:10]}"
            decision = {
                "decision_id": decision_id,
                "status": "OPEN",
                "choice": None,
                "created_at": utc_now_iso(),
                "resolved_at": None,
                "source": "DEFECT_QC",
                "qc_result": self.last_qc_result,
            }
            qc_operator_decisions[decision_id] = decision
            latest_qc_operator_decision_id = decision_id

            notification = {
                "sent_at": utc_now_iso(),
                "status": "sent",
                "channel": "LOCAL_LOG",
                "message": (
                    f"QC FAIL detected. defects={self.last_qc_result.get('defect_count', 0)} "
                    f"changed_ratio={self.last_qc_result.get('changed_ratio', 0):.5f}"
                ),
                "decision_id": decision_id,
            }

            token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
            chat_id = os.getenv("TELEGRAM_OPERATOR_CHAT_ID", "").strip()
            if token and chat_id:
                try:
                    text = (
                        "QC FAIL detected.\n"
                        f"defects={self.last_qc_result.get('defect_count', 0)}\n"
                        f"changed_ratio={self.last_qc_result.get('changed_ratio', 0):.5f}\n"
                        "Choose disposition:"
                    )
                    inline_markup = {
                        "inline_keyboard": [
                            [{"text": "Accept", "callback_data": f"QC|{decision_id}|ACCEPT"}],
                            [{"text": "Discard", "callback_data": f"QC|{decision_id}|DISCARD"}],
                            [{"text": "Rework", "callback_data": f"QC|{decision_id}|REWORK"}],
                        ]
                    }

                    # Prefer sending annotated QC image, fallback to text-only message.
                    sent_photo = False
                    annotated_path = (
                        self.last_qc_artifacts.get("annotated_image")
                        if isinstance(self.last_qc_artifacts, dict)
                        else None
                    )
                    if annotated_path and os.path.exists(annotated_path):
                        with open(annotated_path, "rb") as photo_file:
                            photo_response = requests.post(
                                f"https://api.telegram.org/bot{token}/sendPhoto",
                                data={
                                    "chat_id": chat_id,
                                    "caption": text,
                                    "reply_markup": json.dumps(inline_markup),
                                },
                                files={"photo": photo_file},
                                timeout=12,
                            )
                        photo_response.raise_for_status()
                        photo_result = photo_response.json()
                        if photo_result.get("ok", False):
                            sent_photo = True
                            notification["telegram_photo_sent"] = True

                    if not sent_photo:
                        _telegram_api_call(
                            "sendMessage",
                            {
                                "chat_id": chat_id,
                                "text": text,
                                "reply_markup": inline_markup,
                            },
                        )
                        notification["telegram_photo_sent"] = False

                    notification["channel"] = "TELEGRAM"
                    notification["telegram_ok"] = True
                except Exception as exc:
                    notification["status"] = "failed"
                    notification["channel"] = "TELEGRAM"
                    notification["error"] = str(exc)

            self.last_operator_notification = notification
            if notification["status"] != "sent":
                raise HTTPException(status_code=502, detail=f"Failed to notify operator: {notification.get('error', 'unknown error')}")
            return {
                "ok": True,
                "notification": notification,
                "qc_result": self.last_qc_result,
                "decision": decision,
            }


line_vision_runtime = LineVisionRuntime()
defect_vision_runtime = DefectVisionRuntime()


# ---------------------------------------------------------------------------
# Arke phase sync helper
# ---------------------------------------------------------------------------
def _sync_phase_to_arke(production_order_id: str, operation: str):
    """
    After a local operation is completed, advance the matching phase on Arke.

    1. Fetch the production order from Arke.
    2. Find the phase whose name matches the operation (case-insensitive).
    3. If the phase is not yet started → ``_start`` it.
    4. ``_complete`` the phase.
    """
    try:
        po = arke_client.get(f"/product/production/{production_order_id}")
    except Exception:
        logger.debug(f"Could not fetch PO {production_order_id} from Arke")
        return

    phases = po.get("phases", [])
    op_upper = operation.upper()

    for phase in phases:
        phase_name = (phase.get("phase", {}).get("name") or "").upper()
        if phase_name != op_upper:
            continue

        phase_id = phase.get("id")
        if not phase_id:
            continue

        status = (phase.get("status") or "").lower()

        # Start if not already started
        if status in ("", "draft", "scheduled", "pending"):
            try:
                production_manager.start_phase(phase_id)
                logger.info(f"Arke phase {phase_name} ({phase_id}) started")
            except Exception as e:
                logger.warning(f"Could not start Arke phase {phase_name}: {e}")

        # Complete
        try:
            production_manager.complete_phase(phase_id)
            logger.info(f"Arke phase {phase_name} ({phase_id}) completed")
        except Exception as e:
            logger.warning(f"Could not complete Arke phase {phase_name}: {e}")

        return  # matched phase found, done

    logger.debug(f"No Arke phase matching '{operation}' found on PO {production_order_id}")


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
async def get_production_schedule(
    policy: str = "edf",
    db: Session = Depends(get_db),
):
    """
    Generate production schedule.

    Query params:
        policy: 'edf' (default) | 'group_by_product' | 'split_batches'

    Returns:
        Production schedule with conflict detection
    """
    try:
        from scheduler import SchedulingPolicy

        POLICY_MAP = {
            "edf": SchedulingPolicy.EDF,
            "group_by_product": SchedulingPolicy.GROUP_BY_PRODUCT,
            "split_batches": SchedulingPolicy.SPLIT_BATCHES,
        }
        selected_policy = POLICY_MAP.get(policy, SchedulingPolicy.EDF)

        logger.info(f"Generating production schedule (policy={policy})")
        
        orders = arke_client.get("/sales/order/_active")
        sales_orders = scheduler.parse_sales_orders(orders)
        
        # Create schedule using selected policy
        production_plans = scheduler.create_schedule(sales_orders, selected_policy)
        
        # Detect conflicts
        conflicts = scheduler.detect_conflicts(sales_orders)
        
        # Generate summary
        summary = scheduler.generate_schedule_summary(production_plans, conflicts)
        
        # Build plan dicts (shared by response + persistence)
        plan_dicts = []
        for p in production_plans:
            phase_blocks = scheduler.get_phase_blocks(
                p.product_name, p.quantity, p.starts_at
            )
            plan_dicts.append({
                "order_number": p.sales_order_number,
                "customer": p.customer,
                "product": p.product_name,
                "quantity": p.quantity,
                "priority": p.priority,
                "starts_at": p.starts_at.isoformat(),
                "ends_at": p.ends_at.isoformat(),
                "deadline": p.deadline.isoformat(),
                "reasoning": p.reasoning,
                "phases": phase_blocks,
            })

        # Persist schedule & diff against previous version
        new_version = save_schedule(db, plan_dicts, conflicts)
        
        return {
            "policy": selected_policy.value,
            "generated_at": scheduler.CURRENT_DATE.isoformat(),
            "schedule_version": new_version,
            "production_plans": plan_dicts,
            "conflicts": conflicts,
            "summary": summary,
        }
    except Exception as e:
        logger.error(f"Error generating schedule: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/production-log")
async def get_log(limit: int = 200, db: Session = Depends(get_db)):
    """Return recent production log entries, newest first."""
    entries = get_production_logs(db, limit=limit)
    return [
        {
            "id": e.id,
            "timestamp": e.timestamp.isoformat() if e.timestamp else None,
            "level": e.level,
            "category": e.category,
            "title": e.title,
            "detail": e.detail,
            "order_number": e.order_number,
            "schedule_version": e.schedule_version,
        }
        for e in entries
    ]


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


@app.get("/api/debug/products")
async def debug_products():
    """Debug: show raw product data from Arke to inspect IDs and structure"""
    products = arke_client.get("/product/product")
    # Return first 2 products with all fields for inspection
    return {"count": len(products), "sample": products[:2]}


@app.get("/api/debug/sales-orders")
async def debug_sales_orders():
    """Debug: show raw sales order data to inspect product refs"""
    orders = arke_client.get("/sales/order/_active")
    # Return first order with full structure
    return {"count": len(orders), "sample": orders[:1]}


@app.get("/api/production/push-status")
async def get_production_push_status(db: Session = Depends(get_db)):
    """
    Return push status for every order that has been pushed (or attempted).
    """
    records = get_push_status(db)
    return [
        {
            "order_number": r.order_number,
            "status": r.status,
            "arke_production_order_id": r.arke_production_order_id,
            "error_message": r.error_message,
            "pushed_at": r.pushed_at.isoformat() if r.pushed_at else None,
        }
        for r in records
    ]


@app.post("/api/production/push")
async def push_production_orders(db: Session = Depends(get_db)):
    """
    Push production orders to Arke with idempotency.

    - Skips orders already marked 'pushed'.
    - Retries orders marked 'failed'.
    - Records success/failure per order in ArkePushRecord.
    - Logs every event to the production log.
    """
    try:
        logger.info("Push to Arke triggered")

        # Build the current EDF schedule
        orders = arke_client.get("/sales/order/_active")
        sales_orders = scheduler.parse_sales_orders(orders)
        production_plans = scheduler.create_edf_schedule(sales_orders)

        results = []
        skipped = 0
        created = 0
        failed = 0

        for plan in production_plans:
            order_no = plan.sales_order_number

            # ── Idempotency check ──
            existing = get_push_record(db, order_no)
            if existing and existing.status == "pushed":
                results.append({
                    "order_number": order_no,
                    "status": "skipped",
                    "arke_production_order_id": existing.arke_production_order_id,
                })
                skipped += 1
                continue

            # ── Mark as pushing ──
            upsert_push_record(db, order_no, status="pushing")

            try:
                # Step 3: Create production order in Arke
                production_order = production_manager.create_production_order(plan)
                arke_po_id = production_order.get("id", "")

                # Step 4: Schedule phases (generate from BOM)
                scheduled = production_manager.schedule_phases(arke_po_id)

                # Step 4b: Assign concrete dates to each phase
                production_manager.assign_phase_dates(scheduled, plan)

                # Step 5: Confirm the production order (draft → in_progress)
                try:
                    production_manager.confirm_production_order(arke_po_id)
                    logger.info(f"Confirmed PO {arke_po_id}")
                except Exception as confirm_err:
                    logger.warning(f"Could not confirm PO {arke_po_id}: {confirm_err}")

                # ── Record success ──
                upsert_push_record(db, order_no, status="pushed",
                                   arke_id=arke_po_id, error=None)
                add_production_log(
                    db, level="info", category="push",
                    title=f"{order_no} pushed to Arke",
                    detail=f"Production order {arke_po_id} created and phases scheduled",
                    order_number=order_no,
                )

                results.append({
                    "order_number": order_no,
                    "status": "pushed",
                    "arke_production_order_id": arke_po_id,
                })
                created += 1

            except Exception as order_err:
                error_msg = str(order_err)
                logger.error(f"Failed to push {order_no}: {error_msg}")

                upsert_push_record(db, order_no, status="failed",
                                   error=error_msg)
                add_production_log(
                    db, level="error", category="push",
                    title=f"{order_no} push failed",
                    detail=error_msg,
                    order_number=order_no,
                )

                results.append({
                    "order_number": order_no,
                    "status": "failed",
                    "error": error_msg,
                })
                failed += 1

        summary = f"Pushed {created}, skipped {skipped}, failed {failed}"
        logger.info(summary)

        return {
            "summary": summary,
            "created": created,
            "skipped": skipped,
            "failed": failed,
            "results": results,
        }

    except Exception as e:
        logger.error(f"Error in push to Arke: {str(e)}")
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


@app.post("/api/vision/defect/notify-operator")
async def notify_operator_for_defect_fail():
    return defect_vision_runtime.notify_operator()


@app.get("/api/vision/defect/operator-decision/latest")
async def get_latest_operator_decision():
    if latest_qc_operator_decision_id is None:
        return {"decision": None}
    return {"decision": qc_operator_decisions.get(latest_qc_operator_decision_id)}


@app.get("/api/vision/defect/operator-decisions")
async def get_operator_decisions(limit: int = 20):
    safe_limit = min(max(limit, 1), 200)
    ordered = sorted(
        qc_operator_decisions.values(),
        key=lambda item: item.get("created_at", ""),
        reverse=True,
    )
    return {"count": len(ordered), "decisions": ordered[:safe_limit]}


# ---------------------------------------------------------------------------
# Schedule approval state
# ---------------------------------------------------------------------------
_schedule_approval: Dict[str, Any] = {}


@app.post("/api/telegram/send-schedule")
async def send_schedule_to_telegram(db: Session = Depends(get_db)):
    """
    Send the current schedule to the Telegram operator for approval.

    Posts a formatted summary + inline APPROVE / REJECT buttons.
    Stores the pending approval in ``_schedule_approval`` so the webhook
    can match the callback.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_OPERATOR_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise HTTPException(
            status_code=400,
            detail="TELEGRAM_BOT_TOKEN and TELEGRAM_OPERATOR_CHAT_ID must be set.",
        )

    # Build the current schedule
    orders = arke_client.get("/sales/order/_active")
    sales_orders = scheduler.parse_sales_orders(orders)
    plans = scheduler.create_edf_schedule(sales_orders)
    conflicts = scheduler.detect_conflicts(sales_orders)

    # Format message
    lines = ["📋 *PRODUCTION SCHEDULE — EDF*\n"]
    for i, p in enumerate(plans, 1):
        late = " ⚠️LATE" if p.ends_at > p.deadline else ""
        lines.append(
            f"{i}. `{p.sales_order_number}` {p.product_name} ×{p.quantity}"
            f"  P{p.priority}  {p.starts_at.strftime('%b %-d')}–{p.ends_at.strftime('%b %-d')}{late}"
        )
    if conflicts:
        lines.append(f"\n⚠️ {len(conflicts)} conflict(s):")
        for c in conflicts:
            lines.append(f"• {c['resolution']}")

    text = "\n".join(lines)

    approval_id = f"SCHED-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    _schedule_approval[approval_id] = {
        "id": approval_id,
        "status": "PENDING",
        "created_at": utc_now_iso(),
        "plan_count": len(plans),
    }

    _telegram_api_call("sendMessage", {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "reply_markup": {
            "inline_keyboard": [[
                {"text": "✅ Approve", "callback_data": f"SCHED|{approval_id}|APPROVE"},
                {"text": "❌ Reject", "callback_data": f"SCHED|{approval_id}|REJECT"},
            ]]
        },
    })

    add_production_log(
        db, "info", "telegram",
        f"Schedule sent to Telegram for approval ({approval_id})",
        detail=f"{len(plans)} orders, {len(conflicts)} conflicts",
    )

    return {"ok": True, "approval_id": approval_id, "orders": len(plans)}


@app.get("/api/telegram/schedule-approval")
async def get_schedule_approval_status():
    """Return current schedule approval state."""
    return _schedule_approval


@app.post("/api/telegram/webhook")
async def telegram_webhook(
    payload: TelegramWebhookRequest,
    x_telegram_bot_api_secret_token: Optional[str] = Header(default=None),
):
    secret = os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()
    if secret and x_telegram_bot_api_secret_token != secret:
        raise HTTPException(status_code=403, detail="Invalid Telegram webhook secret.")

    callback_query = payload.callback_query
    if not callback_query:
        return {"ok": True, "handled": False}

    callback_data = str(callback_query.get("data", ""))

    # ── Schedule approval callbacks ──
    if callback_data.startswith("SCHED|"):
        parts = callback_data.split("|")
        if len(parts) == 3:
            _, approval_id, choice = parts
            choice = choice.upper()
            approval = _schedule_approval.get(approval_id)
            if approval and approval["status"] == "PENDING":
                approval["status"] = choice  # APPROVE or REJECT
                approval["resolved_at"] = utc_now_iso()

                # If approved → push + confirm all orders
                if choice == "APPROVE":
                    try:
                        import asyncio
                        # Call the push endpoint internally
                        from database import get_db as _get_db_gen
                        db_gen = _get_db_gen()
                        db = next(db_gen)
                        try:
                            # Reuse the push logic
                            orders_data = arke_client.get("/sales/order/_active")
                            sales_orders = scheduler.parse_sales_orders(orders_data)
                            production_plans = scheduler.create_edf_schedule(sales_orders)
                            for plan in production_plans:
                                order_no = plan.sales_order_number
                                existing_rec = get_push_record(db, order_no)
                                if existing_rec and existing_rec.status == "pushed":
                                    continue
                                upsert_push_record(db, order_no, status="pushing")
                                try:
                                    po = production_manager.create_production_order(plan)
                                    arke_po_id = po.get("id", "")
                                    scheduled = production_manager.schedule_phases(arke_po_id)
                                    production_manager.assign_phase_dates(scheduled, plan)
                                    try:
                                        production_manager.confirm_production_order(arke_po_id)
                                    except Exception:
                                        pass
                                    upsert_push_record(db, order_no, status="pushed",
                                                       arke_id=arke_po_id, error=None)
                                except Exception as push_err:
                                    upsert_push_record(db, order_no, status="failed",
                                                       error=str(push_err))
                            add_production_log(db, "info", "telegram",
                                               f"Schedule approved via Telegram ({approval_id})",
                                               detail="Orders pushed and confirmed")
                        finally:
                            db.close()
                    except Exception as e:
                        logger.error(f"Auto-push after approval failed: {e}")

                # Acknowledge
                try:
                    _telegram_api_call("answerCallbackQuery", {
                        "callback_query_id": callback_query.get("id"),
                        "text": f"Schedule {choice}D",
                        "show_alert": False,
                    })
                    chat_id = callback_query.get("message", {}).get("chat", {}).get("id")
                    if chat_id:
                        emoji = "✅" if choice == "APPROVE" else "❌"
                        _telegram_api_call("sendMessage", {
                            "chat_id": chat_id,
                            "text": f"{emoji} Schedule {choice.lower()}d. "
                                    + ("Orders pushed to Arke." if choice == "APPROVE" else ""),
                        })
                except Exception:
                    pass

            return {"ok": True, "handled": True, "approval": approval}

    # ── QC decision callbacks ──
    if not callback_data.startswith("QC|"):
        return {"ok": True, "handled": False}

    parts = callback_data.split("|")
    if len(parts) != 3:
        return {"ok": True, "handled": False}
    _, decision_id, choice = parts
    choice = choice.upper()
    if choice not in {"ACCEPT", "DISCARD", "REWORK"}:
        return {"ok": True, "handled": False}

    decision = qc_operator_decisions.get(decision_id)
    if not decision:
        return {"ok": True, "handled": False}

    if decision["status"] == "OPEN":
        decision["status"] = "RESOLVED"
        decision["choice"] = choice
        decision["resolved_at"] = utc_now_iso()
        decision["operator"] = {
            "user_id": str(callback_query.get("from", {}).get("id", "")),
            "chat_id": str(callback_query.get("message", {}).get("chat", {}).get("id", "")),
        }

    # Best-effort acknowledgement back to Telegram.
    try:
        _telegram_api_call(
            "answerCallbackQuery",
            {
                "callback_query_id": callback_query.get("id"),
                "text": f"Recorded: {choice}",
                "show_alert": False,
            },
        )
        chat_id = callback_query.get("message", {}).get("chat", {}).get("id")
        if chat_id:
            _telegram_api_call(
                "sendMessage",
                {
                    "chat_id": chat_id,
                    "text": f"QC decision recorded: {choice} (id={decision_id})",
                },
            )
    except Exception:
        pass

    return {"ok": True, "handled": True, "decision": decision}


@app.get("/api/vision/defect/frame")
async def get_defect_vision_frame():
    image_bytes = defect_vision_runtime.frame()
    return Response(content=image_bytes, media_type="image/jpeg")


@app.get("/api/vision/defect/reference-image")
async def get_defect_reference_image():
    path = defect_vision_runtime._absolute_reference_path()
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
    Also syncs with Arke: starts + completes the matching phase.
    """
    try:
        state = finish_operation(db, production_order_id, operation)

        # ── Arke sync: advance the matching phase ──
        try:
            _sync_phase_to_arke(production_order_id, operation)
        except Exception as sync_err:
            logger.warning(f"Arke phase sync failed for {operation}: {sync_err}")

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
