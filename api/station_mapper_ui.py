import argparse
import json
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


STATIONS = ["SMT", "Reflow", "THT", "AOI", "Test", "Coating", "Pack"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Station polygon mapper for live production-line camera.")
    parser.add_argument("--source", default="0", help="Camera source (index like 0/1 or stream URL).")
    parser.add_argument(
        "--config",
        default="../demo_data/station_zones.json",
        help="Path to station polygon config JSON.",
    )
    parser.add_argument("--width", type=int, default=960, help="Canvas width.")
    parser.add_argument("--height", type=int, default=540, help="Canvas height.")
    return parser.parse_args()


class StationMapperApp:
    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self.root = root
        self.root.title("Station Mapper UI")
        self.canvas_width = args.width
        self.canvas_height = args.height
        self.config_path = args.config

        self.cap = None
        self.current_source = args.source
        self.current_frame = None
        self.last_frame_ts = 0.0
        self._frame_lock = threading.Lock()
        self._capture_running = False
        self._capture_thread = None

        self.polygons = []
        self.current_points = []
        self.selected_index = None

        self._build_ui()
        self._connect_source(self.current_source)
        self._load_config_silent()
        self._tick()

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True)

        left = ttk.Frame(container)
        left.pack(side="left", fill="both", expand=True)

        right = ttk.Frame(container, width=320)
        right.pack(side="right", fill="y")

        self.canvas = tk.Canvas(left, width=self.canvas_width, height=self.canvas_height, bg="black")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas_image_id = self.canvas.create_image(0, 0, anchor=tk.NW)

        top_controls = ttk.Frame(right)
        top_controls.pack(fill="x", padx=10, pady=10)

        ttk.Label(top_controls, text="Live Camera Source").pack(anchor="w")
        self.source_var = tk.StringVar(value=self.current_source)
        source_entry = ttk.Entry(top_controls, textvariable=self.source_var)
        source_entry.pack(fill="x", pady=4)

        ttk.Button(top_controls, text="Connect", command=self._connect_clicked).pack(fill="x")

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=10, pady=10)

        ttk.Label(right, text="Assign Station").pack(anchor="w", padx=10)
        self.station_var = tk.StringVar(value=STATIONS[0])
        self.station_dropdown = ttk.Combobox(
            right,
            textvariable=self.station_var,
            values=STATIONS,
            state="readonly",
        )
        self.station_dropdown.current(0)
        self.station_dropdown.pack(fill="x", padx=10, pady=6)

        buttons = ttk.Frame(right)
        buttons.pack(fill="x", padx=10, pady=4)
        ttk.Button(buttons, text="Finish Polygon", command=self._finish_polygon).pack(fill="x", pady=3)
        ttk.Button(buttons, text="Undo Last Point", command=self._undo_point).pack(fill="x", pady=3)
        ttk.Button(buttons, text="Clear Current", command=self._clear_current).pack(fill="x", pady=3)

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=10, pady=10)

        ttk.Label(right, text="Saved Polygons").pack(anchor="w", padx=10)
        self.listbox = tk.Listbox(right, height=12)
        self.listbox.pack(fill="x", padx=10, pady=6)
        self.listbox.bind("<<ListboxSelect>>", self._on_select_polygon)

        list_buttons = ttk.Frame(right)
        list_buttons.pack(fill="x", padx=10)
        ttk.Button(list_buttons, text="Delete Selected", command=self._delete_selected).pack(fill="x", pady=3)

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=10, pady=10)

        io_buttons = ttk.Frame(right)
        io_buttons.pack(fill="x", padx=10)
        ttk.Button(io_buttons, text="Save Config", command=self._save_config).pack(fill="x", pady=3)
        ttk.Button(io_buttons, text="Load Config", command=self._load_config).pack(fill="x", pady=3)
        ttk.Button(io_buttons, text="Save As...", command=self._save_as).pack(fill="x", pady=3)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(right, textvariable=self.status_var, foreground="#1e6").pack(anchor="w", padx=10, pady=10)

    def _status(self, text: str) -> None:
        self.status_var.set(text)

    def _parse_source(self, value: str):
        value = value.strip()
        if value.isdigit():
            return int(value)
        return value

    def _connect_clicked(self) -> None:
        source = self.source_var.get().strip()
        self._connect_source(source)

    def _connect_source(self, source: str) -> None:
        self._stop_capture()
        parsed_source = self._parse_source(source)
        cap = cv2.VideoCapture(parsed_source)
        if not cap.isOpened():
            self._status(f"Failed to open source: {source}")
            messagebox.showerror("Camera Error", f"Could not open source:\n{source}")
            return
        self.cap = cap
        self.current_source = source
        self._start_capture()
        self._status(f"Connected: {source}")

    def _tick(self) -> None:
        self._render()
        self.root.after(30, self._tick)

    def _start_capture(self) -> None:
        if self.cap is None:
            return
        self._capture_running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def _stop_capture(self) -> None:
        self._capture_running = False
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=1.0)
            self._capture_thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _capture_loop(self) -> None:
        while self._capture_running and self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            resized = cv2.resize(frame, (self.canvas_width, self.canvas_height))
            with self._frame_lock:
                self.current_frame = resized
                self.last_frame_ts = time.time()

    def _on_canvas_click(self, event) -> None:
        x, y = int(event.x), int(event.y)
        if 0 <= x < self.canvas_width and 0 <= y < self.canvas_height:
            self.current_points.append((x, y))
            self._render()

    def _undo_point(self) -> None:
        if self.current_points:
            self.current_points.pop()
            self._render()

    def _clear_current(self) -> None:
        self.current_points = []
        self._render()

    def _finish_polygon(self) -> None:
        if len(self.current_points) < 3:
            self._status("Need at least 3 points.")
            return
        station = self.station_var.get()
        norm_points = [(x / self.canvas_width, y / self.canvas_height) for x, y in self.current_points]
        self.polygons.append(
            {
                "station": station,
                "points_norm": norm_points,
            }
        )
        self.current_points = []
        self.selected_index = len(self.polygons) - 1
        self._refresh_listbox()
        self._render()
        self._status(f"Added polygon for {station}.")

    def _on_select_polygon(self, _event=None) -> None:
        selected = self.listbox.curselection()
        self.selected_index = selected[0] if selected else None
        self._render()

    def _delete_selected(self) -> None:
        if self.selected_index is None or self.selected_index >= len(self.polygons):
            return
        del self.polygons[self.selected_index]
        self.selected_index = None
        self._refresh_listbox()
        self._render()
        self._status("Deleted selected polygon.")

    def _refresh_listbox(self) -> None:
        self.listbox.delete(0, tk.END)
        for idx, item in enumerate(self.polygons):
            self.listbox.insert(tk.END, f"{idx + 1}. {item['station']} ({len(item['points_norm'])} pts)")

    def _draw_polygons(self, frame_bgr):
        for idx, item in enumerate(self.polygons):
            points = [
                (int(px * self.canvas_width), int(py * self.canvas_height))
                for px, py in item["points_norm"]
            ]
            if len(points) < 3:
                continue
            color = (0, 255, 0) if idx != self.selected_index else (0, 255, 255)
            contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame_bgr, [contour], True, color, 2)
            label_xy = points[0]
            cv2.putText(
                frame_bgr,
                item["station"],
                (label_xy[0] + 4, label_xy[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    def _draw_current_polygon(self, frame_bgr):
        if not self.current_points:
            return
        pts = np.array(self.current_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame_bgr, [pts], False, (255, 200, 0), 2)
        for x, y in self.current_points:
            cv2.circle(frame_bgr, (x, y), 4, (255, 200, 0), -1)

    def _render(self) -> None:
        with self._frame_lock:
            frame_copy = self.current_frame.copy() if self.current_frame is not None else None
            frame_ts = self.last_frame_ts

        if frame_copy is None:
            blank = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
            frame = blank
        else:
            frame = frame_copy

        self._draw_polygons(frame)
        self._draw_current_polygon(frame)

        age_ms = int((time.time() - frame_ts) * 1000) if frame_ts else -1
        cv2.putText(
            frame,
            f"source={self.current_source} frame_age_ms={age_ms}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"current_station={self.station_var.get()}  current_points={len(self.current_points)}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=image)
        self.canvas.itemconfig(self.canvas_image_id, image=photo)
        self.canvas.image = photo

    def _save_payload(self) -> dict:
        return {
            "version": 1,
            "stations": STATIONS,
            "camera_source": self.current_source,
            "canvas_size": {"width": self.canvas_width, "height": self.canvas_height},
            "polygons": self.polygons,
        }

    def _save_config(self) -> None:
        os.makedirs(os.path.dirname(self.config_path) or ".", exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self._save_payload(), f, indent=2)
        self._status(f"Saved: {self.config_path}")

    def _save_as(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save Station Config",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
        )
        if not path:
            return
        self.config_path = path
        self._save_config()

    def _load_config_silent(self) -> None:
        if not os.path.exists(self.config_path):
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.polygons = payload.get("polygons", [])
            self._refresh_listbox()
            self._render()
            self._status(f"Loaded: {self.config_path}")
        except Exception:
            pass

    def _load_config(self) -> None:
        path = filedialog.askopenfilename(
            title="Load Station Config",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.polygons = payload.get("polygons", [])
            self.config_path = path
            self.selected_index = None
            self.current_points = []
            self._refresh_listbox()
            self._render()
            self._status(f"Loaded: {path}")
        except Exception as exc:
            messagebox.showerror("Load Error", f"Failed to load config:\n{exc}")


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    app = StationMapperApp(root, args)

    def on_close():
        app._stop_capture()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
