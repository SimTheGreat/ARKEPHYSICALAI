import argparse
import os
import time

import cv2
import numpy as np


DEFAULT_CAMERA_INDEX = 0
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30
DEFAULT_ARUCO_DICT = "DICT_5X5_100"
DEFAULT_ARUCO_IDS = "10,11,12,13"  # top-left, top-right, bottom-right, bottom-left
DEFAULT_LOCK_FRAMES = 5
DEFAULT_OUTPUT_DIR = "../demo_data/pcb_images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ArUco-only live debug view.")
    parser.add_argument("--camera-index", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--aruco-dict", default=DEFAULT_ARUCO_DICT)
    parser.add_argument("--aruco-ids", default=DEFAULT_ARUCO_IDS)
    parser.add_argument("--lock-frames", type=int, default=DEFAULT_LOCK_FRAMES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def parse_ids(raw_ids: str) -> list[int]:
    return [int(value.strip()) for value in raw_ids.split(",") if value.strip()]


def open_camera(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    candidates = [index, 1, 2, 3]
    tried = []
    for candidate in candidates:
        if candidate in tried:
            continue
        tried.append(candidate)
        cap = cv2.VideoCapture(candidate)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        # Minimize queue depth where backend supports it.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"Using camera index {candidate}")
        return cap
    raise RuntimeError(f"Could not open camera from indices: {tried}")


def detect_markers(frame: np.ndarray, dict_name: str) -> tuple[list[np.ndarray], np.ndarray | None]:
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(frame)
    return corners, ids


def marker_center(marker_corners: np.ndarray) -> np.ndarray:
    return marker_corners.reshape(4, 2).mean(axis=0)


def marker_outer_corner(marker_corners: np.ndarray, board_center: np.ndarray) -> np.ndarray:
    corners = marker_corners.reshape(4, 2)
    distances = np.linalg.norm(corners - board_center, axis=1)
    return corners[int(np.argmax(distances))]


def expand_quad(quad: np.ndarray, pad_ratio: float = 0.05) -> np.ndarray:
    center = quad.mean(axis=0)
    expanded = center + (quad - center) * (1.0 + pad_ratio)
    return expanded.astype(np.float32)


def warp_from_markers(
    frame: np.ndarray,
    markers_by_id: dict[int, np.ndarray],
    expected_ids: list[int],
    pad_ratio: float = 0.05,
) -> np.ndarray | None:
    if not all(marker_id in markers_by_id for marker_id in expected_ids):
        return None

    marker_centers = np.array([marker_center(markers_by_id[marker_id]) for marker_id in expected_ids], dtype=np.float32)
    board_center = marker_centers.mean(axis=0)
    src = np.array(
        [marker_outer_corner(markers_by_id[marker_id], board_center) for marker_id in expected_ids],
        dtype=np.float32,
    )
    src = expand_quad(src, pad_ratio=pad_ratio)

    tl, tr, br, bl = src
    top_w = np.linalg.norm(tr - tl)
    bottom_w = np.linalg.norm(br - bl)
    left_h = np.linalg.norm(bl - tl)
    right_h = np.linalg.norm(br - tr)
    out_w = int(max(top_w, bottom_w))
    out_h = int(max(left_h, right_h))
    if out_w < 40 or out_h < 40:
        return None
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, matrix, (out_w, out_h))


def main() -> None:
    args = parse_args()
    expected_ids = parse_ids(args.aruco_ids)
    cap = open_camera(args.camera_index, args.width, args.height, args.fps)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.normpath(os.path.join(script_dir, args.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    prev_centers: dict[int, np.ndarray] = {}
    stable_lock_count = 0
    last_ts = time.time()
    frozen = False
    frozen_frame: np.ndarray | None = None
    frozen_warp: np.ndarray | None = None
    frozen_found_ids: list[int] = []

    print("ArUco debug started.")
    print("Keys: [q] quit, [r] reset lock, [s] save current view.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            found_ids = []
            missing_ids = expected_ids.copy()
            jitter_text_parts = []
            display = frame

            if not frozen:
                corners, ids = detect_markers(frame, args.aruco_dict)

                detected: dict[int, np.ndarray] = {}
                if ids is not None and len(ids) > 0:
                    for marker_corners, marker_id in zip(corners, ids.flatten().tolist()):
                        marker = marker_corners.astype(np.float32)
                        detected[marker_id] = marker

                found_ids = sorted([marker_id for marker_id in detected if marker_id in expected_ids])
                missing_ids = [marker_id for marker_id in expected_ids if marker_id not in found_ids]

                full_lock = len(found_ids) == len(expected_ids)
                if full_lock:
                    stable_lock_count += 1
                else:
                    stable_lock_count = 0

                if ids is not None and len(ids) > 0:
                    cv2.aruco.drawDetectedMarkers(display, corners, ids)

                centers: dict[int, np.ndarray] = {}
                for marker_id in expected_ids:
                    if marker_id not in detected:
                        continue
                    center = marker_center(detected[marker_id])
                    centers[marker_id] = center
                    if marker_id in prev_centers:
                        jitter = float(np.linalg.norm(center - prev_centers[marker_id]))
                        jitter_text_parts.append(f"{marker_id}:{jitter:.1f}px")
                    prev_centers[marker_id] = center

                    cx, cy = int(center[0]), int(center[1])
                    cv2.circle(display, (cx, cy), 4, (0, 255, 255), -1)
                    cv2.putText(
                        display,
                        f"ID {marker_id}",
                        (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                if stable_lock_count >= args.lock_frames and len(found_ids) == len(expected_ids):
                    frozen = True
                    frozen_found_ids = found_ids
                    frozen_frame = display.copy()
                    frozen_warp = warp_from_markers(frame, detected, expected_ids, pad_ratio=0.06)
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    raw_path = os.path.join(output_dir, f"aruco_lock_raw_{stamp}.png")
                    cv2.imwrite(raw_path, frozen_frame)
                    print(f"Lock acquired. Saved raw snapshot: {raw_path}")
                    if frozen_warp is not None:
                        warp_path = os.path.join(output_dir, f"aruco_lock_warp_{stamp}.png")
                        cv2.imwrite(warp_path, frozen_warp)
                        print(f"Saved warped snapshot: {warp_path}")
            else:
                display = frozen_frame.copy() if frozen_frame is not None else frame
                found_ids = frozen_found_ids
                missing_ids = [marker_id for marker_id in expected_ids if marker_id not in found_ids]

            now = time.time()
            dt = max(now - last_ts, 1e-6)
            fps = 1.0 / dt
            last_ts = now

            status = "FROZEN" if frozen else ("LOCKED" if stable_lock_count >= args.lock_frames else "SEARCHING")
            status_color = (0, 200, 0) if status == "LOCKED" else (0, 0, 255)
            if status == "FROZEN":
                status_color = (0, 220, 0)

            cv2.putText(display, f"Status: {status}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2, cv2.LINE_AA)
            cv2.putText(
                display,
                f"Found IDs: {found_ids} Missing: {missing_ids}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                f"FPS: {fps:.1f}",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                f"Jitter(px): {' | '.join(jitter_text_parts) if jitter_text_parts else 'n/a'}",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            if frozen:
                cv2.putText(
                    display,
                    "Detection frozen. Press 'r' to unlock and search again.",
                    (20, 155),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 180),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("ArUco Debug", display)
            if frozen_warp is not None:
                cv2.imshow("ArUco Locked Warp", frozen_warp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                frozen = False
                frozen_frame = None
                frozen_warp = None
                frozen_found_ids = []
                stable_lock_count = 0
                prev_centers = {}
                print("Lock reset.")
            if key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join(output_dir, f"aruco_debug_{stamp}.png")
                cv2.imwrite(path, display)
                print(f"Saved snapshot: {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
