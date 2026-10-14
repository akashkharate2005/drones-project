"""Standalone face detection and gallery-matching demonstration."""
import argparse
import glob
import os
import time

import cv2
import numpy as np

from src.perception.faces import FaceRecognizer
from src.utils.config import load_config


def draw_results(frame, results, backend):
    """Draw recognition results and the active backend on a frame."""
    output = frame.copy()
    for match in results:
        x1, y1, x2, y2 = match.box
        color = (0, 180, 0) if match.is_authorized else (0, 0, 220)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{match.name} | {match.confidence:.2f}"
        cv2.putText(
            output,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(output, (0, 0), (output.shape[1], 42), (25, 25, 25), -1)
    cv2.putText(
        output,
        f"FACE INSPECTION DEMO | Backend: {backend} | Press Q to quit",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def fit_to_tile(frame, tile_width, tile_height, background=(20, 20, 20)):
    """Resize a frame to fit completely inside a tile without distortion."""
    height, width = frame.shape[:2]
    scale = min(tile_width / width, tile_height / height)
    resized = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))
    tile = np.full((tile_height, tile_width, 3), background, dtype=np.uint8)
    x_offset = (tile_width - resized.shape[1]) // 2
    y_offset = (tile_height - resized.shape[0]) // 2
    tile[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized
    return tile


def add_image_title(tile, image_path, index, total):
    """Add a readable title without covering the detected-image content."""
    title = f"{index}/{total}  {os.path.basename(image_path)}"
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 32), (25, 25, 25), -1)
    cv2.putText(tile, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def print_results(image_path, results, backend):
    print(f"[FaceDemo] Backend: {backend}")
    print(f"[FaceDemo] Image: {image_path}")
    if results:
        for result in results:
            print(
                f"[FaceDemo] {result.name} | authorized={result.is_authorized} "
                f"| confidence={result.confidence:.3f}"
            )
    else:
        print("[FaceDemo] No face detected.")


def run_image(recognizer, image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    results = recognizer.recognize_faces(frame)
    print_results(image_path, results, recognizer.backend)

    display = draw_results(frame, results, recognizer.backend)
    while True:
        cv2.imshow("Face Detection Demo", display)
        if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


def run_test_gallery(recognizer, test_dir):
    image_paths = sorted(
        path for pattern in ("*.jpg", "*.jpeg", "*.png")
        for path in glob.glob(os.path.join(test_dir, pattern))
    )
    if not image_paths:
        raise FileNotFoundError(f"No test images found in: {test_dir}")

    print(f"[FaceDemo] Found {len(image_paths)} test image(s).")
    print("[FaceDemo] Showing all test images in a fitted grid. Press Q/Esc to quit.")
    tiles = []
    tile_width, tile_height = 620, 400
    for index, image_path in enumerate(image_paths, start=1):
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[FaceDemo] Skipping unreadable image: {image_path}")
            continue
        results = recognizer.recognize_faces(frame)
        print(f"\n[FaceDemo] Test image {index}/{len(image_paths)}")
        print_results(image_path, results, recognizer.backend)
        annotated = draw_results(frame, results, recognizer.backend)
        tile = fit_to_tile(annotated, tile_width, tile_height)
        tiles.append(add_image_title(tile, image_path, index, len(image_paths)))

    columns = min(2, len(tiles))
    rows = (len(tiles) + columns - 1) // columns
    grid = np.full((rows * tile_height, columns * tile_width, 3), (10, 10, 10), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        grid[row * tile_height:(row + 1) * tile_height,
             column * tile_width:(column + 1) * tile_width] = tile

    cv2.imshow("Face Detection Demo - Test Gallery", grid)
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


def run_webcam(recognizer, camera_id):
    camera = cv2.VideoCapture(camera_id)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open webcam {camera_id}.")

    print("[FaceDemo] Live mode started. Press Q or Esc to quit.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("The webcam stopped returning frames.")
            results = recognizer.recognize_faces(frame)
            cv2.imshow("Face Detection Demo", draw_results(frame, results, recognizer.backend))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            time.sleep(0.01)
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Show face detection and gallery matching.")
    parser.add_argument(
        "--image",
        default="",
        help="Still image to inspect instead of the test-image gallery.",
    )
    parser.add_argument("--test-dir", default="data/images/test", help="Directory of multi-person test images.")
    parser.add_argument("--gallery-dir", default="data/images", help="Directory containing labeled person folders.")
    parser.add_argument(
        "--people",
        nargs="+",
        default=["Akash", "Laksh", "Ritu"],
        help="Labeled gallery folders to use for identification.",
    )
    parser.add_argument("--webcam", action="store_true", help="Use a live webcam instead of a still image.")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam index for --webcam.")
    args = parser.parse_args()

    config = load_config()
    config["perception"]["faces"]["authorized_dir"] = args.gallery_dir
    config["perception"]["faces"]["authorized_names"] = args.people
    config["perception"]["faces"]["unknown_dir"] = os.path.join(args.gallery_dir, "__no_unknown_gallery__")
    recognizer = FaceRecognizer(config)
    if recognizer.backend in ("opencv", "opencv_simple"):
        raise RuntimeError(
            "An identity-grade backend is unavailable. Install the project dependencies "
            "with `python -m pip install -r requirements.txt`, then rerun this demo. "
            "The OpenCV fallback is intentionally disabled for identity claims."
        )
    if args.webcam:
        run_webcam(recognizer, args.camera_id)
    elif args.image:
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Image does not exist: {args.image}")
        run_image(recognizer, args.image)
    else:
        run_test_gallery(recognizer, args.test_dir)


if __name__ == "__main__":
    main()
