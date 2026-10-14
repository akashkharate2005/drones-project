"""Telemetry, alert, and mission summary logger."""
import csv
import json
import os
import time
from typing import Any, Dict, List, Optional


class MissionLogger:
    """Logs runtime telemetry to CSV and JSON files for post-mission evaluation."""

    def __init__(self, config: Dict[str, Any]):
        log_cfg = config.get("logging", {})
        self.output_dir = log_cfg.get("output_dir", "results")
        os.makedirs(self.output_dir, exist_ok=True)

        self.telemetry_path = os.path.join(self.output_dir, "telemetry.csv")
        self.alerts_path = os.path.join(self.output_dir, "alerts.csv")
        self.summary_path = os.path.join(self.output_dir, "mission_summary.json")

        self.start_time = time.time()
        self.telemetry_count = 0
        self.alerts: List[Dict[str, Any]] = []
        self.peak_crowd_count = 0

        # Initialize CSV files with headers
        with open(self.telemetry_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "sim_time", "frame_idx",
                "drone_x", "drone_y", "drone_z", "yaw",
                "battery", "crowd_count", "ground_truth_count",
                "alert_level", "mission_state", "fps"
            ])

        with open(self.alerts_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "sim_time", "alert_level", "trigger_reason", "details"])

    def log_telemetry(
        self,
        sim_time: float,
        frame_idx: int,
        drone_pose: Any,
        battery: float,
        crowd_count: int,
        ground_truth_count: int,
        alert_level: str,
        mission_state: str,
        fps: float
    ) -> None:
        """Append a telemetry record to CSV."""
        self.telemetry_count += 1
        if crowd_count > self.peak_crowd_count:
            self.peak_crowd_count = crowd_count

        with open(self.telemetry_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                f"{time.time() - self.start_time:.3f}",
                f"{sim_time:.3f}",
                frame_idx,
                f"{getattr(drone_pose, 'x', 0.0):.2f}",
                f"{getattr(drone_pose, 'y', 0.0):.2f}",
                f"{getattr(drone_pose, 'z', 0.0):.2f}",
                f"{getattr(drone_pose, 'yaw', 0.0):.2f}",
                f"{battery:.1f}",
                crowd_count,
                ground_truth_count,
                alert_level,
                mission_state,
                f"{fps:.1f}"
            ])

    def log_alert(
        self,
        sim_time: float,
        alert_level: str,
        trigger_reason: str,
        details: str = ""
    ) -> None:
        """Append an alert transition to CSV and internal list."""
        record = {
            "elapsed_sec": round(time.time() - self.start_time, 2),
            "sim_time": round(sim_time, 2),
            "alert_level": alert_level,
            "trigger_reason": trigger_reason,
            "details": details
        }
        self.alerts.append(record)

        with open(self.alerts_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                f"{time.time() - self.start_time:.3f}",
                f"{sim_time:.3f}",
                alert_level,
                trigger_reason,
                details
            ])

    def save_summary(self, extra_meta: Optional[Dict[str, Any]] = None) -> None:
        """Write mission summary to JSON."""
        summary = {
            "mission_duration_sec": round(time.time() - self.start_time, 2),
            "total_frames_logged": self.telemetry_count,
            "peak_crowd_count": self.peak_crowd_count,
            "total_alerts": len(self.alerts),
            "alerts_history": self.alerts,
            "metadata": extra_meta or {}
        }
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
