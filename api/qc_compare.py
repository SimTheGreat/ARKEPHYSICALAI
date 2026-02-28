import argparse
import json
import os
from typing import Dict, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple digital QC comparator.")
    parser.add_argument("--reference", required=True, help="Path to golden/reference image.")
    parser.add_argument("--test", required=True, help="Path to test image.")
    parser.add_argument(
        "--output",
        default="qc_result.png",
        help="Annotated output image path.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=400,
        help="Minimum contour area (pixels) to count as defect.",
    )
    parser.add_argument(
        "--fail-ratio",
        type=float,
        default=0.0015,
        help="Fail if changed pixel ratio exceeds this threshold.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=35,
        help="Binary threshold for absdiff mask.",
    )
    parser.add_argument(
        "--aruco-dict",
        default="DICT_5X5_100",
        help="OpenCV ArUco dictionary name.",
    )
    parser.add_argument(
        "--aruco-ids",
        default="10,11,12,13",
        help="Comma-separated marker IDs expected on the board.",
    )
    parser.add_argument(
        "--no-aruco",
        action="store_true",
        help="Disable ArUco-based alignment/ROI and run plain image diff.",
    )
    return parser.parse_args()


def read_color(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def parse_ids(raw_ids: str) -> list[int]:
    return [int(item.strip()) for item in raw_ids.split(",") if item.strip()]


def detect_aruco_points(
    image_bgr: np.ndarray,
    dictionary_name: str,
    expected_ids: list[int],
) -> Dict[int, np.ndarray]:
    aruco_module = cv2.aruco
    dictionary = aruco_module.getPredefinedDictionary(getattr(aruco_module, dictionary_name))
    detector = aruco_module.ArucoDetector(dictionary, aruco_module.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(image_bgr)
    if ids is None or len(ids) == 0:
        return {}

    found: Dict[int, np.ndarray] = {}
    for marker_corners, marker_id in zip(corners, ids.flatten().tolist()):
        if marker_id in expected_ids:
            found[marker_id] = marker_corners[0].astype(np.float32)
    return found


def align_test_to_reference(
    ref_bgr: np.ndarray,
    test_bgr: np.ndarray,
    dictionary_name: str,
    expected_ids: list[int],
) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[int, np.ndarray], bool]:
    ref_markers = detect_aruco_points(ref_bgr, dictionary_name, expected_ids)
    test_markers = detect_aruco_points(test_bgr, dictionary_name, expected_ids)
    common_ids = [marker_id for marker_id in expected_ids if marker_id in ref_markers and marker_id in test_markers]

    # Need at least 4 point pairs for homography; one marker contributes 4 points.
    if len(common_ids) < 1:
        resized = cv2.resize(test_bgr, (ref_bgr.shape[1], ref_bgr.shape[0]))
        return resized, ref_markers, test_markers, False

    src_pts = np.concatenate([test_markers[marker_id] for marker_id in common_ids], axis=0)
    dst_pts = np.concatenate([ref_markers[marker_id] for marker_id in common_ids], axis=0)

    homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if homography is None:
        resized = cv2.resize(test_bgr, (ref_bgr.shape[1], ref_bgr.shape[0]))
        return resized, ref_markers, test_markers, False

    warped = cv2.warpPerspective(test_bgr, homography, (ref_bgr.shape[1], ref_bgr.shape[0]))
    return warped, ref_markers, test_markers, True


def marker_inner_corner(marker_corners: np.ndarray, board_center: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(marker_corners - board_center, axis=1)
    return marker_corners[np.argmin(distances)]


def build_board_mask_from_markers(image_shape: tuple, ref_markers: Dict[int, np.ndarray], expected_ids: list[int]) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if not all(marker_id in ref_markers for marker_id in expected_ids):
        return mask

    centers = np.array([ref_markers[marker_id].mean(axis=0) for marker_id in expected_ids], dtype=np.float32)
    board_center = centers.mean(axis=0)

    quad = np.array(
        [
            marker_inner_corner(ref_markers[expected_ids[0]], board_center),  # top-left marker inner corner
            marker_inner_corner(ref_markers[expected_ids[1]], board_center),  # top-right
            marker_inner_corner(ref_markers[expected_ids[2]], board_center),  # bottom-right
            marker_inner_corner(ref_markers[expected_ids[3]], board_center),  # bottom-left
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(mask, quad, 255)
    return mask


def main() -> None:
    args = parse_args()
    expected_ids = parse_ids(args.aruco_ids)

    ref_bgr = read_color(args.reference)
    test_bgr = read_color(args.test)

    if args.no_aruco:
        aligned_test = cv2.resize(test_bgr, (ref_bgr.shape[1], ref_bgr.shape[0]))
        ref_markers = {}
        aligned_ok = False
    else:
        aligned_test, ref_markers, _, aligned_ok = align_test_to_reference(
            ref_bgr=ref_bgr,
            test_bgr=test_bgr,
            dictionary_name=args.aruco_dict,
            expected_ids=expected_ids,
        )

    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(aligned_test, cv2.COLOR_BGR2GRAY)

    # Reduce lighting variance before differencing.
    ref_gray = cv2.equalizeHist(ref_gray)
    test_gray = cv2.equalizeHist(test_gray)
    ref_blur = cv2.GaussianBlur(ref_gray, (5, 5), 0)
    test_blur = cv2.GaussianBlur(test_gray, (5, 5), 0)

    diff = cv2.absdiff(ref_blur, test_blur)
    _, mask = cv2.threshold(diff, args.threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    board_mask = build_board_mask_from_markers(ref_bgr.shape, ref_markers, expected_ids)
    if np.count_nonzero(board_mask) > 0:
        mask = cv2.bitwise_and(mask, board_mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ref_preview = ref_bgr.copy()
    annotated = aligned_test.copy()

    defect_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < args.min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        defect_regions.append({"x": x, "y": y, "w": w, "h": h, "area": float(area)})
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)

    changed_pixels = int(np.count_nonzero(mask))
    total_pixels = int(mask.shape[0] * mask.shape[1])
    changed_ratio = changed_pixels / total_pixels if total_pixels else 0.0
    has_large_regions = len(defect_regions) > 0
    qc_status = "FAIL" if has_large_regions or changed_ratio > args.fail_ratio else "PASS"

    status_color = (0, 200, 0) if qc_status == "PASS" else (0, 0, 255)
    cv2.putText(
        annotated,
        f"QC: {qc_status}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        status_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"changed_ratio={changed_ratio:.5f}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"aruco_aligned={aligned_ok}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    if np.count_nonzero(board_mask) > 0:
        board_contours, _ = cv2.findContours(board_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, board_contours, -1, (0, 255, 255), 2)
        cv2.drawContours(ref_preview, board_contours, -1, (0, 255, 255), 2)

    stacked = np.hstack([ref_preview, annotated])
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cv2.imwrite(args.output, stacked)

    result = {
        "status": qc_status,
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "changed_ratio": changed_ratio,
        "defect_count": len(defect_regions),
        "defect_regions": defect_regions,
        "aruco_aligned": aligned_ok,
        "aruco_found_reference": sorted(ref_markers.keys()),
        "output_image": args.output,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
