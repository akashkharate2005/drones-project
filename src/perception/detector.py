"""YOLOv8 Nano person detector, ByteTrack tracker, and spatial crowd density estimator."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class DetectionResult:
    """Encapsulates perception outputs for a single video frame."""
    frame_idx: int
    count: int
    unique_tracked_count: int
    boxes: List[Tuple[int, int, int, int]] = field(default_factory=list) # [x1, y1, x2, y2]
    track_ids: List[int] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    centroids: List[Tuple[float, float]] = field(default_factory=list)
    density_map: np.ndarray = field(default_factory=lambda: np.zeros((10, 10), dtype=np.float32))
    hotspot_centroid: Optional[Tuple[float, float]] = None # (norm_x, norm_y) in [0, 1]
    is_hotspot: bool = False
    peak_density: float = 0.0


class PersonDetector:
    """Detects and tracks persons in aerial frames, estimating density and identifying hotspots."""

    def __init__(self, config: Dict[str, Any]):
        det_cfg = config.get("perception", {}).get("detector", {})
        dens_cfg = config.get("perception", {}).get("density", {})

        self.model_name = det_cfg.get("model_name", "yolov8n.pt")
        self.imgsz = det_cfg.get("imgsz", 960)
        self.conf = det_cfg.get("conf", 0.35)
        self.iou = det_cfg.get("iou", 0.45)
        self.person_class_id = det_cfg.get("person_class_id", 0)
        self.tracker_config = det_cfg.get("tracker_config", "bytetrack.yaml")

        self.grid_rows = dens_cfg.get("grid_rows", 10)
        self.grid_cols = dens_cfg.get("grid_cols", 10)
        self.gaussian_sigma = dens_cfg.get("gaussian_sigma", 1.5)
        self.hotspot_ratio_threshold = dens_cfg.get("hotspot_ratio_threshold", 0.35)

        self.unique_track_ids = set()
        self.frame_counter = 0

        # Lazy load YOLOv8 to allow fast import
        self._model = None

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_name)
        return self._model

    def detect_and_track(self, frame: np.ndarray) -> DetectionResult:
        """Run YOLOv8 inference with ByteTrack on the input frame.

        Args:
            frame: BGR numpy image (H, W, 3).

        Returns:
            DetectionResult containing boxes, track IDs, count, and density heatmap.
        """
        self.frame_counter += 1
        h, w = frame.shape[:2]
        model = self._load_model()

        # Run tracking with ByteTrack, filtering strictly for person class (class 0)
        try:
            results = model.track(
                source=frame,
                persist=True,
                classes=[self.person_class_id],
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                tracker=self.tracker_config,
                verbose=False
            )
        except Exception:
            # Fallback to standard detect if tracker encounters an issue
            results = model(
                source=frame,
                classes=[self.person_class_id],
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                verbose=False
            )

        boxes_out: List[Tuple[int, int, int, int]] = []
        track_ids_out: List[int] = []
        confs_out: List[float] = []
        centroids_out: List[Tuple[float, float]] = []

        if results and len(results) > 0 and results[0].boxes is not None:
            r_boxes = results[0].boxes
            xyxy = r_boxes.xyxy.cpu().numpy() if hasattr(r_boxes.xyxy, "cpu") else r_boxes.xyxy
            confs = r_boxes.conf.cpu().numpy() if hasattr(r_boxes.conf, "cpu") else r_boxes.conf
            t_ids = (
                r_boxes.id.cpu().numpy().astype(int)
                if (r_boxes.id is not None and hasattr(r_boxes.id, "cpu"))
                else (r_boxes.id.astype(int) if r_boxes.id is not None else None)
            )

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
                conf = float(confs[i])
                tid = int(t_ids[i]) if (t_ids is not None and i < len(t_ids)) else (i + 1)

                boxes_out.append((x1, y1, x2, y2))
                track_ids_out.append(tid)
                confs_out.append(conf)
                self.unique_track_ids.add(tid)

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                centroids_out.append((cx, cy))

        # Fallback to aerial blob contrast detector if YOLO finds 0 boxes (e.g. synthetic top-down pedestrians)
        if len(boxes_out) == 0:
            blob_boxes = self._detect_aerial_blobs(frame)
            for i, (bx1, by1, bx2, by2) in enumerate(blob_boxes):
                boxes_out.append((bx1, by1, bx2, by2))
                tid = (i + 1)
                track_ids_out.append(tid)
                confs_out.append(0.72)
                self.unique_track_ids.add(tid)
                centroids_out.append(((bx1 + bx2) / 2.0, (by1 + by2) / 2.0))

        # Compute spatial density grid and heatmap
        density_grid, hotspot_norm, is_hotspot, peak_val = self._compute_density(centroids_out, w, h)

        return DetectionResult(
            frame_idx=self.frame_counter,
            count=len(boxes_out),
            unique_tracked_count=len(self.unique_track_ids),
            boxes=boxes_out,
            track_ids=track_ids_out,
            confidences=confs_out,
            centroids=centroids_out,
            density_map=density_grid,
            hotspot_centroid=hotspot_norm,
            is_hotspot=is_hotspot,
            peak_density=peak_val
        )

    def _detect_aerial_blobs(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Aerial contrast detector identifying small walking figures on ground."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bg = cv2.medianBlur(gray, 25)
        diff = cv2.absdiff(gray, bg)
        _, thresh = cv2.threshold(diff, 16, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if 35 <= area <= 4000:
                x, y, bw, bh = cv2.boundingRect(c)
                if 5 <= bw <= 90 and 5 <= bh <= 90:
                    boxes.append((x, y, x + bw, y + bh))
        return boxes

    def _compute_density(
        self,
        centroids: List[Tuple[float, float]],
        frame_w: int,
        frame_h: int
    ) -> Tuple[np.ndarray, Optional[Tuple[float, float]], bool, float]:
        """Bin centroids into an NxM grid, apply Gaussian smoothing, and locate peak density."""
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        if not centroids:
            return grid, None, False, 0.0

        for cx, cy in centroids:
            col = int(np.clip((cx / max(1, frame_w)) * self.grid_cols, 0, self.grid_cols - 1))
            row = int(np.clip((cy / max(1, frame_h)) * self.grid_rows, 0, self.grid_rows - 1))
            grid[row, col] += 1.0

        raw_peak = float(np.max(grid))
        # Apply Gaussian filter for continuous spatial density field
        smoothed_grid = gaussian_filter(grid, sigma=self.gaussian_sigma)
        peak_val = float(np.max(smoothed_grid))

        hotspot_norm: Optional[Tuple[float, float]] = None
        is_hotspot = False

        if peak_val > 0.0:
            max_idx = np.unravel_index(np.argmax(smoothed_grid), smoothed_grid.shape)
            r_peak, c_peak = int(max_idx[0]), int(max_idx[1])
            norm_x = (c_peak + 0.5) / self.grid_cols
            norm_y = (r_peak + 0.5) / self.grid_rows
            hotspot_norm = (norm_x, norm_y)

            # A hotspot is recognized if a significant crowd concentration exists in a region
            if len(centroids) >= 6 and (raw_peak >= 4 or (raw_peak / len(centroids)) >= self.hotspot_ratio_threshold):
                is_hotspot = True

        return smoothed_grid, hotspot_norm, is_hotspot, peak_val

    def reset_tracker(self):
        """Reset unique IDs and frame counter."""
        self.unique_track_ids.clear()
        self.frame_counter = 0
