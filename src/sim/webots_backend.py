"""Webots EXTERN controller simulator backend for DJI Mavic 2 Pro."""
import os
import sys
import time
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from src.sim.base import SimulatorBackend
from src.control.flight import DronePose, MotorCommands


class WebotsBackend(SimulatorBackend):
    """Communicates directly with Webots via EXTERN controller API (WEBOTS_CONTROLLER_URL)."""

    def __init__(self, config: Dict[str, Any]):
        sim_cfg = config.get("simulation", {})
        self.time_step_ms = sim_cfg.get("time_step_ms", 32)
        self.controller_url = sim_cfg.get("webots_controller_url", "ipc://1270/mavic_2_pro")

        self.robot = None
        self.camera = None
        self.gps = None
        self.imu = None
        self.gyro = None
        self.motors = []
        self.camera_pitch_motor = None

        self.battery = 100.0
        self.battery_drain_rate = config.get("decision", {}).get("battery_drain_per_sec", 0.06)

        self.last_frame = np.zeros((720, 960, 3), dtype=np.uint8)
        self.current_pose = DronePose()
        self.ground_truth_pedestrians: List[Dict[str, Any]] = []

        self._ensure_webots_python_path()

    def _ensure_webots_python_path(self) -> None:
        """Locates Webots installation directory and appends lib/controller/python to sys.path."""
        candidate_paths = [
            os.environ.get("WEBOTS_HOME", ""),
            r"C:\Program Files\Webots",
            r"C:\Users\khara\AppData\Local\Programs\Webots",
            r"/usr/local/webots",
            r"/Applications/Webots.app"
        ]
        for base in candidate_paths:
            if base and os.path.exists(base):
                py_path = os.path.join(base, "lib", "controller", "python")
                if os.path.exists(py_path) and py_path not in sys.path:
                    sys.path.append(py_path)
                    break

    def connect(self) -> bool:
        """Connects to the Webots simulation process."""
        try:
            # Set environment variable for EXTERN controller connection
            if "WEBOTS_CONTROLLER_URL" not in os.environ and self.controller_url:
                os.environ["WEBOTS_CONTROLLER_URL"] = self.controller_url

            from controller import Robot, Supervisor

            # Use Supervisor to enable reading ground-truth pedestrian coordinates
            try:
                self.robot = Supervisor()
            except Exception:
                self.robot = Robot()

            self.time_step_ms = int(self.robot.getBasicTimeStep())
            print(f"[WebotsBackend] Connected to Webots robot '{self.robot.getName()}', timestep: {self.time_step_ms}ms")

            # Initialize Camera
            self.camera = self.robot.getDevice("camera")
            if self.camera:
                self.camera.enable(self.time_step_ms)

            # Initialize GPS and IMU
            self.gps = self.robot.getDevice("gps")
            if self.gps:
                self.gps.enable(self.time_step_ms)

            self.imu = self.robot.getDevice("inertial unit")
            if self.imu:
                self.imu.enable(self.time_step_ms)

            self.gyro = self.robot.getDevice("gyro")
            if self.gyro:
                self.gyro.enable(self.time_step_ms)

            # Initialize 4 rotor motors
            motor_names = [
                "front left propeller",
                "front right propeller",
                "rear left propeller",
                "rear right propeller"
            ]
            self.motors = []
            for name in motor_names:
                m = self.robot.getDevice(name)
                if m:
                    m.setPosition(float('inf'))
                    m.setVelocity(0.0)
                    self.motors.append(m)

            self.camera_pitch_motor = self.robot.getDevice("camera pitch")
            if self.camera_pitch_motor:
                self.camera_pitch_motor.setPosition(-1.57) # Straight down

            return True

        except Exception as e:
            print(f"[WebotsBackend] Failed to connect to Webots: {e}")
            print("                Ensure Webots is running with the world open and robot controller set to <extern>.")
            return False

    def step(self, dt: float) -> bool:
        """Advance Webots simulation by one time step."""
        if self.robot is None:
            return False

        status = self.robot.step(self.time_step_ms)
        if status == -1:
            return False

        # Drain battery over time
        self.battery = max(0.0, self.battery - (self.battery_drain_rate * dt))

        # Update pose from GPS & IMU
        if self.gps and self.imu:
            gps_val = self.gps.getValues()
            rpy = self.imu.getRollPitchYaw()
            gyro_val = self.gyro.getValues() if self.gyro else [0.0, 0.0, 0.0]

            self.current_pose = DronePose(
                x=gps_val[0],
                y=gps_val[1],
                z=gps_val[2],
                roll=rpy[0],
                pitch=rpy[1],
                yaw=rpy[2],
                yaw_rate=gyro_val[2]
            )

        # Update ground truth pedestrians if supervisor mode is active
        self._update_ground_truth_pedestrians()

        return True

    def get_frame(self) -> np.ndarray:
        """Extract camera image from Webots shared memory buffer."""
        if self.camera is None:
            return self.last_frame

        raw_image = self.camera.getImage()
        if raw_image is None:
            return self.last_frame

        w = self.camera.getWidth()
        h = self.camera.getHeight()
        # Webots camera buffer format: BGRA
        bgra = np.frombuffer(raw_image, np.uint8).reshape((h, w, 4))
        self.last_frame = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        return self.last_frame

    def get_pose(self) -> DronePose:
        return self.current_pose

    def send_command(self, cmd: MotorCommands) -> None:
        """Set rotational velocities on the 4 propeller motors and pitch on the gimbal."""
        if len(self.motors) == 4:
            self.motors[0].setVelocity(cmd.m1_front_left)
            self.motors[1].setVelocity(cmd.m2_front_right)
            self.motors[2].setVelocity(cmd.m3_rear_left)
            self.motors[3].setVelocity(cmd.m4_rear_right)

        if self.camera_pitch_motor is not None:
            self.camera_pitch_motor.setPosition(cmd.gimbal_pitch)

    def get_battery(self) -> float:
        return self.battery

    def get_ground_truth_pedestrians(self) -> List[Dict[str, Any]]:
        return self.ground_truth_pedestrians

    def _update_ground_truth_pedestrians(self) -> None:
        """Queries supervisor nodes to track known pedestrian coordinates in world."""
        if not hasattr(self.robot, "getFromDef"):
            return

        gt_list = []
        for i in range(1, 60):
            node = self.robot.getFromDef(f"PEDESTRIAN_{i}")
            if node is not None:
                pos = node.getPosition()
                gt_list.append({"id": i, "x": pos[0], "y": pos[1], "z": pos[2]})
        self.ground_truth_pedestrians = gt_list

    def disconnect(self) -> None:
        if self.robot is not None:
            print("[WebotsBackend] Disconnecting from Webots.")
            self.robot = None
