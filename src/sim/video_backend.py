"""Video file and synthetic generator backend simulating flight kinematics and camera frames."""
import math
import os
import time
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from src.sim.base import SimulatorBackend
from src.control.flight import DronePose, MotorCommands
from src.utils.synthetic_feed import SyntheticFeedGenerator


class VideoFileBackend(SimulatorBackend):
    """Simulates a drone payload flight by reading video frames or generating

    live synthetic plaza surveillance scenes while updating 6-DOF flight kinematics.
    """

    def __init__(self, config: Dict[str, Any], video_path: Optional[str] = None):
        self.config = config
        self.video_path = video_path

        cam_cfg = config.get("camera", {})
        self.width = cam_cfg.get("width", 960)
        self.height = cam_cfg.get("height", 720)

        fl_cfg = config.get("flight", {})
        self.patrol_alt = fl_cfg.get("patrol_altitude", 28.0)
        self.battery_drain_rate = config.get("decision", {}).get("battery_drain_per_sec", 0.06)

        self.cap: Optional[cv2.VideoCapture] = None
        self.generator = SyntheticFeedGenerator(width=self.width, height=self.height, num_pedestrians=42)

        # Simulated 6-DOF flight kinematics
        self.pose = DronePose(x=0.0, y=0.0, z=0.5, roll=0.0, pitch=0.0, yaw=0.0)
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = self.patrol_alt
        self.target_yaw = 0.0

        self.battery = 100.0
        self.current_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.ground_truth_pedestrians: List[Dict[str, Any]] = []
        self.is_connected = False
        self.cluster_event_active = False

    def connect(self) -> bool:
        if self.video_path and os.path.exists(self.video_path):
            self.cap = cv2.VideoCapture(self.video_path)
            print(f"[VideoFileBackend] Loaded surveillance video from: {self.video_path}")
        else:
            print("[VideoFileBackend] Using dynamic flight-synchronized synthetic plaza renderer.")
            self.cap = None

        self.is_connected = True
        return True

    def step(self, dt: float = 0.05) -> bool:
        if not self.is_connected:
            return False

        # Drain battery
        self.battery = max(0.0, self.battery - (self.battery_drain_rate * dt))

        # Update simulated drone flight physics smoothly toward targets
        self.pose.z += (self.target_z - self.pose.z) * min(1.0, 2.5 * dt)
        self.pose.x += (self.target_x - self.pose.x) * min(1.0, 1.8 * dt)
        self.pose.y += (self.target_y - self.pose.y) * min(1.0, 1.8 * dt)
        self.pose.yaw += (self.target_yaw - self.pose.yaw) * min(1.0, 2.0 * dt)

        # Retrieve next frame
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                # Loop video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()

            if ret and frame is not None:
                if frame.shape[:2] != (self.height, self.width):
                    frame = cv2.resize(frame, (self.width, self.height))
                self.current_frame = frame
                # Approximate ground truth count based on current synthetic model
                self.ground_truth_pedestrians = [{"id": i, "x": 0.0, "y": 0.0} for i in range(25)]
                return True

        # Render live synthetic frame corresponding to simulated drone pose
        self.generator.update_pedestrians(dt=dt, cluster_mode=self.cluster_event_active)
        self.current_frame, self.ground_truth_pedestrians = self.generator.generate_frame(
            drone_x=self.pose.x,
            drone_y=self.pose.y,
            drone_z=self.pose.z,
            drone_yaw=self.pose.yaw
        )

        return True

    def get_frame(self) -> np.ndarray:
        return self.current_frame

    def get_pose(self) -> DronePose:
        return self.pose

    def send_command(self, cmd: MotorCommands) -> None:
        """Actuation feedback: infer tilt and altitude velocity from rotor outputs."""
        thrust_net = (cmd.m1_front_left - cmd.m2_front_right - cmd.m3_rear_left + cmd.m4_rear_right) / 4.0
        vz = (thrust_net - 68.5) * 0.15
        self.pose.vz = vz

    def set_flight_targets(self, tx: float, ty: float, tz: float, tyaw: float = 0.0) -> None:
        """Direct setpoint hook for flight controller synchronization."""
        self.target_x = tx
        self.target_y = ty
        self.target_z = tz
        self.target_yaw = tyaw

    def trigger_hotspot_event(self, active: bool = True) -> None:
        """Induces synthetic crowd clustering to test hotspot alert workflows."""
        self.cluster_event_active = active

    def get_battery(self) -> float:
        return self.battery

    def get_ground_truth_pedestrians(self) -> List[Dict[str, Any]]:
        return self.ground_truth_pedestrians

    def disconnect(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_connected = False
