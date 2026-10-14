"""Webcam backend capturing live USB camera frames for physical demonstration."""
from typing import Any, Dict, List
import cv2
import numpy as np

from src.sim.base import SimulatorBackend
from src.control.flight import DronePose, MotorCommands


class WebcamBackend(SimulatorBackend):
    """Enables running the AI surveillance payload directly on a physical USB camera."""

    def __init__(self, config: Dict[str, Any], camera_index: int = 0):
        self.camera_index = camera_index
        cam_cfg = config.get("camera", {})
        self.width = cam_cfg.get("width", 960)
        self.height = cam_cfg.get("height", 720)

        self.cap = None
        self.battery = 98.0
        self.battery_drain_rate = config.get("decision", {}).get("battery_drain_per_sec", 0.05)
        self.current_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.pose = DronePose(x=0.0, y=0.0, z=28.0, roll=0.0, pitch=0.0, yaw=0.0)

    def connect(self) -> bool:
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"[WebcamBackend] Warning: Could not open camera index {self.camera_index}. Using blank frames.")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        print(f"[WebcamBackend] Live camera connected on index {self.camera_index}.")
        return True

    def step(self, dt: float = 0.05) -> bool:
        self.battery = max(0.0, self.battery - (self.battery_drain_rate * dt))

        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                if frame.shape[:2] != (self.height, self.width):
                    frame = cv2.resize(frame, (self.width, self.height))
                self.current_frame = frame
                return True

        return True

    def get_frame(self) -> np.ndarray:
        return self.current_frame

    def get_pose(self) -> DronePose:
        return self.pose

    def send_command(self, cmd: MotorCommands) -> None:
        pass

    def get_battery(self) -> float:
        return self.battery

    def get_ground_truth_pedestrians(self) -> List[Dict[str, Any]]:
        return []

    def disconnect(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
