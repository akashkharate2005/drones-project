"""Abstract base class and contract for all simulation and hardware backends."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np

from src.control.flight import DronePose, MotorCommands


class SimulatorBackend(ABC):
    """Abstract interface for sensor acquisition and flight actuation."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to simulator or hardware."""
        pass

    @abstractmethod
    def step(self, dt: float) -> bool:
        """Advance simulation or sensor step. Returns False if simulation has ended."""
        pass

    @abstractmethod
    def get_frame(self) -> np.ndarray:
        """Retrieve latest camera frame as BGR uint8 numpy array."""
        pass

    @abstractmethod
    def get_pose(self) -> DronePose:
        """Retrieve current 6-DOF estimated drone pose."""
        pass

    @abstractmethod
    def send_command(self, cmd: MotorCommands) -> None:
        """Send actuator commands (motor speeds, gimbal angle) to drone."""
        pass

    @abstractmethod
    def get_battery(self) -> float:
        """Retrieve current battery level percentage [0 - 100]."""
        pass

    @abstractmethod
    def get_ground_truth_pedestrians(self) -> List[Dict[str, Any]]:
        """Retrieve ground-truth pedestrian world and image positions if available."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Gracefully release resources and disconnect."""
        pass
