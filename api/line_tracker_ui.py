import argparse
import json
import os
import threading
import time
import tkinter as tk
from collections import deque
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


DEFAULT_CONFIG_PATH = "../demo_data/station_zones.json"
DEFAULT_SOURCE = "0"
DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 540
DEFAULT_LOWER_HSV = (20, 80, 80)
DEFAULT_UPPER_HSV = (40, 255, 255)
DEFAULT_MIN_AREA = 800
DEFAULT_SCORE_DECAY = 0.92
DEFAULT_SWITCH_SCORE = 0.55
DEFAULT_DEBOUNCE_FRAMES = 5
DEFAULT_TRAIL_LEN = 40
DEFAULT_PROFILE_PATH = "../demo_data/line_tracker_profile.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live production line station tracker.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Station polygon JSON from station_mapper_ui.py")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Camera source index (0/1/2) or stream URL")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--min-area", type=int, default=DEFAULT_MIN_AREA, help="Minimum contour area for tracked box")
    parser.add_argument("--score-decay", type=float, default=DEFAULT_SCORE_DECAY, help="Temporal station score decay")
    parser.add_argument("--switch-score", type=float, default=DEFAULT_SWITCH_SCORE, help="Min dominant score to allow station switch")
    parser.add_argument("--debounce-frames", type=int, default=DEFAULT_DEBOUNCE_FRAMES, help="Frames needed to confirm station switch")
    parser.add_argument("--trail-len", type=int, default=DEFAULT_TRAIL_LEN, help="Centroid trail length")
    return parser.parse_args()


def parse_source(raw: str):
    raw = raw.strip()
    return int(raw) if raw.isdigit() else raw


class StationTrackerUI:
    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self.root = root
        self.args = args
        self.root.title("Line Station Tracker")

        self.canvas_w = args.width
        self.canvas_h = args.height
        self.cap = None
        self._capture_running = False
        self._capture_thread = None
        self._frame_lock = threading.Lock()
        self.current_frame = None
        self.frame_ts = 0.0

        self.station_polygons = self._load_station_polygons(args.config)
        if not self.station_polygons:
            raise RuntimeError(f"No polygons found in config: {args.config}")

        stations = sorted({item["station"] for item in self.station_polygons})
        self.station_scores = {station: 0.0 for station in stations}
        self.current_station = "NOT_PRESENT"
        self.candidate_station = None
        self.candidate_count = 0
        self.centroid_trail = deque(maxlen=args.trail_len)
        self.last_transition = None
        self.last_detection_area = 0.0
        self.instant_candidates = []

        self.lower_h = tk.IntVar(value=DEFAULT_LOWER_HSV[0])
        self.lower_s = tk.IntVar(value=DEFAULT_LOWER_HSV[1])
        self.lower_v = tk.IntVar(value=DEFAULT_LOWER_HSV[2])
        self.upper_h = tk.IntVar(value=DEFAULT_UPPER_HSV[0])
        self.upper_s = tk.IntVar(value=DEFAULT_UPPER_HSV[1])
        self.upper_v = tk.IntVar(value=DEFAULT_UPPER_HSV[2])

        self.status_var = tk.StringVar(value="Starting...")
        self.station_var = tk.StringVar(value="NOT_PRESENT")
        self.score_var = tk.StringVar(value="{}")
        self.profile_path = DEFAULT_PROFILE_PATH

        self._build_ui()
        self._load_profile_silent()
        self._connect(args.source)
        self._tick()

    def _load_station_polygons(self, config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        polygons = data.get("polygons", [])
        normalized = []
        for item in polygons:
            station = item.get("station")
            points_norm = item.get("points_norm", [])
            if not station or len(points_norm) < 3:
                continue
            normalized.append({"station": station, "points_norm": points_norm})
        return normalized

    def _build_ui(self):
        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True)

        left = ttk.Frame(container)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(container, width=340)
        right.pack(side="right", fill="y")

        self.canvas = tk.Canvas(left, width=self.canvas_w, height=self.canvas_h, bg="black")
        self.canvas.pack(fill="both", expand=True)
        self.canvas_image_id = self.canvas.create_image(0, 0, anchor=tk.NW)

        ttk.Label(right, text="Source").pack(anchor="w", padx=10, pady=(10, 2))
        self.source_var = tk.StringVar(value=self.args.source)
        ttk.Entry(right, textvariable=self.source_var).pack(fill="x", padx=10)
        ttk.Button(right, text="Connect", command=self._on_connect).pack(fill="x", padx=10, pady=6)

        ttk.Label(right, text="Detected Station").pack(anchor="w", padx=10, pady=(8, 2))
        ttk.Label(right, textvariable=self.station_var, font=("Helvetica", 16, "bold")).pack(anchor="w", padx=10)
        ttk.Label(right, textvariable=self.score_var, wraplength=320).pack(anchor="w", padx=10, pady=(4, 8))

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=10, pady=8)
        ttk.Label(right, text="HSV Thresholds").pack(anchor="w", padx=10)
        self._add_slider(right, "Lower H", self.lower_h, 0, 179)
        self._add_slider(right, "Lower S", self.lower_s, 0, 255)
        self._add_slider(right, "Lower V", self.lower_v, 0, 255)
        self._add_slider(right, "Upper H", self.upper_h, 0, 179)
        self._add_slider(right, "Upper S", self.upper_s, 0, 255)
        self._add_slider(right, "Upper V", self.upper_v, 0, 255)

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=10, pady=8)
        ttk.Label(right, text="Tracker Profile").pack(anchor="w", padx=10)
        ttk.Button(right, text="Save Profile", command=self._save_profile).pack(fill="x", padx=10, pady=3)
        ttk.Button(right, text="Load Profile", command=self._load_profile).pack(fill="x", padx=10, pady=3)

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=10, pady=8)
        ttk.Label(right, textvariable=self.status_var, wraplength=320).pack(anchor="w", padx=10, pady=6)

    def _add_slider(self, parent, label, var, minv, maxv):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=10, pady=2)
        ttk.Label(row, text=label, width=8).pack(side="left")
        tk.Scale(row, from_=minv, to=maxv, orient="horizontal", variable=var, length=220).pack(side="left")

    def _status(self, text: str):
        self.status_var.set(text)

    def _connect(self, source: str):
        self._stop_capture()
        cap = cv2.VideoCapture(parse_source(source))
        if not cap.isOpened():
            messagebox.showerror("Camera Error", f"Cannot open source: {source}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.canvas_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.canvas_h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap = cap
        self._capture_running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        self._status(f"Connected source: {source}")

    def _on_connect(self):
        self._connect(self.source_var.get().strip())

    def _stop_capture(self):
        self._capture_running = False
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=1.0)
            self._capture_thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _capture_loop(self):
        while self._capture_running and self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = cv2.resize(frame, (self.canvas_w, self.canvas_h))
            with self._frame_lock:
                self.current_frame = frame
                self.frame_ts = time.time()

    def _normalized_polygons_px(self):
        result = []
        for poly in self.station_polygons:
            pts = np.array(
                [(int(x * self.canvas_w), int(y * self.canvas_h)) for x, y in poly["points_norm"]],
                dtype=np.int32,
            )
            result.append({"station": poly["station"], "points": pts})
        return result

    def _detect_centroid(self, frame_bgr):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([self.lower_h.get(), self.lower_s.get(), self.lower_v.get()], dtype=np.uint8)
        upper = np.array([self.upper_h.get(), self.upper_s.get(), self.upper_v.get()], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0, mask

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < self.args.min_area:
            return None, area, mask

        m = cv2.moments(contour)
        if m["m00"] == 0:
            return None, area, mask
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        return (cx, cy), area, mask

    def _candidate_stations(self, centroid, polygons_px):
        if centroid is None:
            return []
        candidates = []
        for item in polygons_px:
            inside = cv2.pointPolygonTest(item["points"], centroid, False) >= 0
            if inside:
                candidates.append(item["station"])
        return candidates

    def _update_station_scores(self, candidates):
        for station in self.station_scores:
            self.station_scores[station] *= self.args.score_decay

        if not candidates:
            self.candidate_station = None
            self.candidate_count = 0
            self.current_station = "NOT_PRESENT"
            return

        if candidates:
            # If centroid overlaps multiple polygons, distribute one vote across them.
            inc = 1.0 / len(candidates)
            for station in candidates:
                self.station_scores[station] += inc

        dominant_station = max(self.station_scores, key=lambda s: self.station_scores[s])
        dominant_score = self.station_scores[dominant_station]

        if dominant_score < self.args.switch_score:
            return

        if dominant_station != self.current_station:
            if self.candidate_station == dominant_station:
                self.candidate_count += 1
            else:
                self.candidate_station = dominant_station
                self.candidate_count = 1
            if self.candidate_count >= self.args.debounce_frames:
                prev = self.current_station
                self.current_station = dominant_station
                self.last_transition = (prev, dominant_station, time.strftime("%H:%M:%S"))
                self.candidate_station = None
                self.candidate_count = 0
        else:
            self.candidate_station = None
            self.candidate_count = 0

    def _draw_overlay(self, frame_bgr, polygons_px, centroid):
        for item in polygons_px:
            station = item["station"]
            pts = item["points"].reshape((-1, 1, 2))
            is_current = station == self.current_station
            color = (0, 255, 0) if is_current else (255, 200, 0)
            cv2.polylines(frame_bgr, [pts], True, color, 2)
            p0 = item["points"][0]
            cv2.putText(frame_bgr, station, (int(p0[0]) + 4, int(p0[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        if centroid is not None:
            self.centroid_trail.append(centroid)
            for i, pt in enumerate(self.centroid_trail):
                intensity = int(50 + 205 * (i + 1) / max(1, len(self.centroid_trail)))
                cv2.circle(frame_bgr, pt, 2, (intensity, 0, 255 - intensity), -1)
            cv2.circle(frame_bgr, centroid, 6, (0, 0, 255), -1)

    def _tick(self):
        with self._frame_lock:
            frame = self.current_frame.copy() if self.current_frame is not None else None
            frame_ts = self.frame_ts

        if frame is not None:
            polygons_px = self._normalized_polygons_px()
            centroid, area, mask = self._detect_centroid(frame)
            self.last_detection_area = area
            candidates = self._candidate_stations(centroid, polygons_px)
            self.instant_candidates = candidates
            self._update_station_scores(candidates)
            self._draw_overlay(frame, polygons_px, centroid)

            age_ms = int((time.time() - frame_ts) * 1000) if frame_ts > 0 else -1
            cv2.putText(frame, f"frame_age_ms={age_ms}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"instant_candidates={candidates if candidates else ['NOT_PRESENT']}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"current_station={self.current_station}",
                (10, 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(image=image)
            self.canvas.itemconfig(self.canvas_image_id, image=photo)
            self.canvas.image = photo

            score_summary = ", ".join(f"{k}:{v:.2f}" for k, v in sorted(self.station_scores.items()))
            self.station_var.set(self.current_station)
            self.score_var.set(score_summary)
            transition = f" last_transition={self.last_transition}" if self.last_transition else ""
            self._status(f"area={self.last_detection_area:.0f} instant={self.instant_candidates or ['NOT_PRESENT']}{transition}")

        self.root.after(30, self._tick)

    def close(self):
        self._stop_capture()

    def _profile_payload(self) -> dict:
        return {
            "version": 1,
            "source": self.source_var.get().strip(),
            "hsv": {
                "lower_h": self.lower_h.get(),
                "lower_s": self.lower_s.get(),
                "lower_v": self.lower_v.get(),
                "upper_h": self.upper_h.get(),
                "upper_s": self.upper_s.get(),
                "upper_v": self.upper_v.get(),
            },
            "tracker": {
                "min_area": self.args.min_area,
                "score_decay": self.args.score_decay,
                "switch_score": self.args.switch_score,
                "debounce_frames": self.args.debounce_frames,
                "trail_len": self.args.trail_len,
            },
            "config_path": self.args.config,
        }

    def _apply_profile(self, payload: dict) -> None:
        hsv = payload.get("hsv", {})
        if "lower_h" in hsv:
            self.lower_h.set(int(hsv["lower_h"]))
        if "lower_s" in hsv:
            self.lower_s.set(int(hsv["lower_s"]))
        if "lower_v" in hsv:
            self.lower_v.set(int(hsv["lower_v"]))
        if "upper_h" in hsv:
            self.upper_h.set(int(hsv["upper_h"]))
        if "upper_s" in hsv:
            self.upper_s.set(int(hsv["upper_s"]))
        if "upper_v" in hsv:
            self.upper_v.set(int(hsv["upper_v"]))

        tracker = payload.get("tracker", {})
        if "min_area" in tracker:
            self.args.min_area = int(tracker["min_area"])
        if "score_decay" in tracker:
            self.args.score_decay = float(tracker["score_decay"])
        if "switch_score" in tracker:
            self.args.switch_score = float(tracker["switch_score"])
        if "debounce_frames" in tracker:
            self.args.debounce_frames = int(tracker["debounce_frames"])
        if "trail_len" in tracker:
            self.args.trail_len = int(tracker["trail_len"])
            self.centroid_trail = deque(self.centroid_trail, maxlen=self.args.trail_len)

        source = payload.get("source")
        if source:
            self.source_var.set(str(source))

    def _save_profile(self) -> None:
        initial_dir = os.path.dirname(self.profile_path) or "."
        initial_file = os.path.basename(self.profile_path) or "line_tracker_profile.json"
        path = filedialog.asksaveasfilename(
            title="Save Tracker Profile",
            initialdir=initial_dir,
            initialfile=initial_file,
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
        )
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._profile_payload(), f, indent=2)
        self.profile_path = path
        self._status(f"Profile saved: {path}")

    def _load_profile(self) -> None:
        initial_dir = os.path.dirname(self.profile_path) or "."
        path = filedialog.askopenfilename(
            title="Load Tracker Profile",
            initialdir=initial_dir,
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self._apply_profile(payload)
            self.profile_path = path
            self._status(f"Profile loaded: {path}")
        except Exception as exc:
            messagebox.showerror("Profile Error", f"Failed to load profile:\n{exc}")

    def _load_profile_silent(self) -> None:
        if not os.path.exists(self.profile_path):
            return
        try:
            with open(self.profile_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self._apply_profile(payload)
            self._status(f"Profile loaded: {self.profile_path}")
        except Exception:
            pass


def main():
    args = parse_args()
    root = tk.Tk()
    app = StationTrackerUI(root, args)

    def on_close():
        app.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
