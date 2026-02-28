import argparse
import json
import os
import threading
import time

import cv2
import numpy as np

from qc_compare import (
    build_board_mask_from_markers,
    detect_aruco_points,
    parse_ids,
    read_color,
)

DEFAULT_REFERENCE_PATH = "../demo_data/pcb_images/pcb_ref_image.png"
DEFAULT_CAMERA_INDEX = 0
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30
DEFAULT_MIN_AREA = 800
DEFAULT_FAIL_RATIO = 0.003
DEFAULT_THRESHOLD = 45
DEFAULT_ARUCO_DICT = "DICT_5X5_100"
DEFAULT_ARUCO_IDS = "10,11,12,13"
DEFAULT_OUTPUT_DIR = "../demo_data/pcb_images/live_outputs"
DEFAULT_SMOOTH_ALPHA = 0.25
DEFAULT_MAX_JUMP_PX = 22.0
DEFAULT_FAIL_HYSTERESIS = 3
DEFAULT_PASS_HYSTERESIS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live QC detector using webcam + ArUco alignment.")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE_PATH, help="Path to golden/reference image.")
    parser.add_argument("--camera-index", type=int, default=DEFAULT_CAMERA_INDEX, help="OpenCV camera index (try 0,1,2).")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Requested capture height.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Requested FPS.")
    parser.add_argument("--min-area", type=int, default=DEFAULT_MIN_AREA, help="Minimum defect contour area.")
    parser.add_argument("--fail-ratio", type=float, default=DEFAULT_FAIL_RATIO, help="Fail threshold for changed ratio.")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD, help="Binary threshold for absdiff mask.")
    parser.add_argument("--aruco-dict", default=DEFAULT_ARUCO_DICT, help="OpenCV ArUco dictionary name.")
    parser.add_argument("--aruco-ids", default=DEFAULT_ARUCO_IDS, help="Marker IDs: top-left,top-right,bottom-right,bottom-left.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for saved snapshots.")
    parser.add_argument("--smooth-alpha", type=float, default=DEFAULT_SMOOTH_ALPHA, help="Marker EMA smoothing factor.")
    parser.add_argument("--max-jump-px", type=float, default=DEFAULT_MAX_JUMP_PX, help="Reject marker jumps above this px distance.")
    parser.add_argument("--fail-hysteresis", type=int, default=DEFAULT_FAIL_HYSTERESIS, help="Frames required to switch PASS->FAIL.")
    parser.add_argument("--pass-hysteresis", type=int, default=DEFAULT_PASS_HYSTERESIS, help="Frames required to switch FAIL->PASS.")
    return parser.parse_args()


def open_camera(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    candidates = [index, 1, 2, 3]
    tried = []
    for cam_index in candidates:
        if cam_index in tried:
            continue
        tried.append(cam_index)
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        print(f"Using camera index {cam_index}")
        return cap
    raise RuntimeError(f"Could not open any camera from indices {tried}.")


class FrameGrabber:
    """Continuously grabs frames and keeps only the latest to avoid stale-buffer freezes."""

    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_ts = 0.0
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def get_latest(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            if self._latest_frame is None:
                return None, self._latest_ts
            return self._latest_frame.copy(), self._latest_ts

    def _run(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._latest_frame = frame
                self._latest_ts = time.time()


class TemporalAligner:
    def __init__(
        self,
        ref_bgr: np.ndarray,
        expected_ids: list[int],
        aruco_dict: str,
        smooth_alpha: float,
        max_jump_px: float,
    ):
        self.ref_bgr = ref_bgr
        self.expected_ids = expected_ids
        self.aruco_dict = aruco_dict
        self.smooth_alpha = smooth_alpha
        self.max_jump_px = max_jump_px
        self.ref_markers = detect_aruco_points(ref_bgr, aruco_dict, expected_ids)
        self.filtered_markers: dict[int, np.ndarray] = {}
        self.prev_homography: np.ndarray | None = None

    def align(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, bool, dict[int, np.ndarray], dict[int, np.ndarray]]:
        detected = detect_aruco_points(frame_bgr, self.aruco_dict, self.expected_ids)
        current_filtered: dict[int, np.ndarray] = {}

        for marker_id in self.expected_ids:
            if marker_id not in detected:
                if marker_id in self.filtered_markers:
                    current_filtered[marker_id] = self.filtered_markers[marker_id]
                continue

            marker = detected[marker_id]
            if marker_id in self.filtered_markers:
                prev = self.filtered_markers[marker_id]
                jump = float(np.mean(np.linalg.norm(marker - prev, axis=1)))
                if jump > self.max_jump_px:
                    marker = prev
                else:
                    marker = (self.smooth_alpha * marker + (1.0 - self.smooth_alpha) * prev).astype(np.float32)

            current_filtered[marker_id] = marker.astype(np.float32)

        self.filtered_markers = current_filtered

        common_ids = [
            marker_id
            for marker_id in self.expected_ids
            if marker_id in self.ref_markers and marker_id in current_filtered
        ]

        aligned_ok = False
        if len(common_ids) >= 1:
            src_pts = np.concatenate([current_filtered[marker_id] for marker_id in common_ids], axis=0)
            dst_pts = np.concatenate([self.ref_markers[marker_id] for marker_id in common_ids], axis=0)
            homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
            if homography is not None:
                self.prev_homography = homography
                aligned_ok = True

        if not aligned_ok and self.prev_homography is not None:
            homography = self.prev_homography
            aligned_ok = True

        if aligned_ok:
            aligned = cv2.warpPerspective(frame_bgr, homography, (self.ref_bgr.shape[1], self.ref_bgr.shape[0]))
        else:
            aligned = cv2.resize(frame_bgr, (self.ref_bgr.shape[1], self.ref_bgr.shape[0]))

        return aligned, aligned_ok, detected, current_filtered


def compute_qc(
    ref_bgr: np.ndarray,
    frame_bgr: np.ndarray,
    aligner: TemporalAligner,
    board_mask: np.ndarray,
    threshold: int,
    min_area: int,
    fail_ratio: float,
) -> dict:
    aligned, aligned_ok, detected_markers, filtered_markers = aligner.align(frame_bgr)

    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

    ref_gray = cv2.equalizeHist(ref_gray)
    test_gray = cv2.equalizeHist(test_gray)
    ref_blur = cv2.GaussianBlur(ref_gray, (5, 5), 0)
    test_blur = cv2.GaussianBlur(test_gray, (5, 5), 0)

    diff = cv2.absdiff(ref_blur, test_blur)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    if np.count_nonzero(board_mask) > 0:
        mask = cv2.bitwise_and(mask, board_mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defect_regions = []
    annotated = aligned.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        defect_regions.append({"x": x, "y": y, "w": w, "h": h, "area": float(area)})
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)

    changed_pixels = int(np.count_nonzero(mask))
    total_pixels = int(mask.shape[0] * mask.shape[1])
    changed_ratio = changed_pixels / total_pixels if total_pixels else 0.0
    qc_status = "FAIL" if defect_regions or changed_ratio > fail_ratio else "PASS"

    if np.count_nonzero(board_mask) > 0:
        board_contours, _ = cv2.findContours(board_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, board_contours, -1, (0, 255, 255), 2)

    status_color = (0, 200, 0) if qc_status == "PASS" else (0, 0, 255)
    cv2.putText(annotated, f"QC: {qc_status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2, cv2.LINE_AA)
    cv2.putText(
        annotated,
        f"changed_ratio={changed_ratio:.5f} defects={len(defect_regions)}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        (
            f"aruco_aligned={aligned_ok} "
            f"detected={sorted(detected_markers.keys())} "
            f"filtered={sorted(filtered_markers.keys())}"
        ),
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return {
        "status": qc_status,
        "changed_ratio": changed_ratio,
        "defect_regions": defect_regions,
        "aruco_aligned": aligned_ok,
        "aruco_detected": sorted(detected_markers.keys()),
        "aruco_filtered": sorted(filtered_markers.keys()),
        "annotated": annotated,
    }


def main() -> None:
    args = parse_args()
    expected_ids = parse_ids(args.aruco_ids)
    ref_bgr = read_color(args.reference)
    cap = open_camera(args.camera_index, args.width, args.height, args.fps)
    grabber = FrameGrabber(cap)
    grabber.start()
    os.makedirs(args.output_dir, exist_ok=True)
    aligner = TemporalAligner(
        ref_bgr=ref_bgr,
        expected_ids=expected_ids,
        aruco_dict=args.aruco_dict,
        smooth_alpha=args.smooth_alpha,
        max_jump_px=args.max_jump_px,
    )
    board_mask = build_board_mask_from_markers(ref_bgr.shape, aligner.ref_markers, expected_ids)
    stable_status = "PASS"
    fail_streak = 0
    pass_streak = 0

    print("Live QC started.")
    print("Keys: [q] quit, [s] save current annotated frame + JSON metrics.")

    try:
        while True:
            frame, frame_ts = grabber.get_latest()
            if frame is None:
                time.sleep(0.05)
                continue

            result = compute_qc(
                ref_bgr=ref_bgr,
                frame_bgr=frame,
                aligner=aligner,
                board_mask=board_mask,
                threshold=args.threshold,
                min_area=args.min_area,
                fail_ratio=args.fail_ratio,
            )

            if result["status"] == "FAIL":
                fail_streak += 1
                pass_streak = 0
            else:
                pass_streak += 1
                fail_streak = 0

            if stable_status == "PASS" and fail_streak >= args.fail_hysteresis:
                stable_status = "FAIL"
            elif stable_status == "FAIL" and pass_streak >= args.pass_hysteresis:
                stable_status = "PASS"

            overlay_color = (0, 200, 0) if stable_status == "PASS" else (0, 0, 255)
            cv2.putText(
                result["annotated"],
                f"stable_qc={stable_status} (f={fail_streak},p={pass_streak})",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                overlay_color,
                2,
                cv2.LINE_AA,
            )
            frame_age_ms = int((time.time() - frame_ts) * 1000) if frame_ts > 0 else -1
            cv2.putText(
                result["annotated"],
                f"frame_age_ms={frame_age_ms}",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            view = np.hstack([ref_bgr, result["annotated"]])
            cv2.imshow("Live QC (Reference | Live)", view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                image_path = os.path.join(args.output_dir, f"qc_live_{stamp}.png")
                json_path = os.path.join(args.output_dir, f"qc_live_{stamp}.json")
                cv2.imwrite(image_path, view)
                payload = {
                    "status": result["status"],
                    "stable_status": stable_status,
                    "changed_ratio": result["changed_ratio"],
                    "defect_count": len(result["defect_regions"]),
                    "defect_regions": result["defect_regions"],
                    "aruco_aligned": result["aruco_aligned"],
                    "saved_image": image_path,
                }
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                print(json.dumps(payload, indent=2))
    finally:
        grabber.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
