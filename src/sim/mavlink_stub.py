"""MAVLink / ArduPilot SITL backend stub for physical Pixhawk or SITL flight controllers.

ARCHITECTURE & HARDWARE INTEGRATION NOTES:
------------------------------------------
In the physical hardware payload, an onboard companion computer (Raspberry Pi 4 / Jetson Nano)
interfaces with the Pixhawk flight controller running ArduPilot/PX4 via a TELEM UART port
(e.g., /dev/ttyAMA0 at 921600 baud) or via UDP when communicating with ArduPilot SITL (udp:127.0.0.1:14550).

Data Flow:
1. Companion computer listens for high-rate MAVLink messages:
   - ATTITUDE (msg ID 30): roll, pitch, yaw, rollspeed, pitchspeed, yawspeed
   - GLOBAL_POSITION_INT (msg ID 33): lat, lon, alt, relative_alt, vx, vy, vz, hdg
   - SYS_STATUS (msg ID 1): battery_remaining percentage, voltage, current
2. Flight commands are dispatched via:
   - SET_POSITION_TARGET_LOCAL_NED (msg ID 84): Offboard position / velocity setpoints
   - MAV_CMD_DO_MOUNT_CONTROL (msg ID 205): Gimbal pitch tilt control
   - MAV_CMD_NAV_RETURN_TO_LAUNCH: Automatic hardware RTH
"""
from typing import Any, Dict, List, Optional
import numpy as np

from src.sim.base import SimulatorBackend
from src.control.flight import DronePose, MotorCommands


class MAVLinkBackendStub(SimulatorBackend):
    """Documented stub and integration blueprint for Pixhawk hardware / ArduPilot SITL.

    To enable in hardware:
    1. Install: pip install pymavlink
    2. Set MAVLINK_CONNECTION_STRING (e.g., 'udp:127.0.0.1:14550' or '/dev/ttyAMA0:921600')
    3. Uncomment the pymavlink message listeners in step() and connect()
    """

    def __init__(self, config: Dict[str, Any], connection_string: str = "udp:127.0.0.1:14550"):
        self.connection_string = connection_string
        self.config = config
        self.vehicle = None
        self.pose = DronePose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0)
        self.battery = 100.0
        self.is_connected = False
        self.current_frame = np.zeros((720, 960, 3), dtype=np.uint8)

    def connect(self) -> bool:
        """Connects to Pixhawk / SITL via pymavlink."""
        print(f"[MAVLinkBackend] Connecting to flight controller at {self.connection_string}...")
        try:
            # Example hardware initialization:
            # from pymavlink import mavutil
            # self.vehicle = mavutil.mavlink_connection(self.connection_string)
            # self.vehicle.wait_heartbeat(timeout=5)
            # print(f"[MAVLinkBackend] Heartbeat received from system {self.vehicle.target_system}")
            self.is_connected = True
            return True
        except Exception as e:
            print(f"[MAVLinkBackend] Connection failed: {e}")
            return False

    def step(self, dt: float = 0.05) -> bool:
        """Poll incoming MAVLink messages and update pose."""
        if not self.is_connected:
            return False

        # In live pymavlink deployment:
        # msg = self.vehicle.recv_match(type=['ATTITUDE', 'GLOBAL_POSITION_INT', 'SYS_STATUS'], blocking=False)
        # while msg is not None:
        #     if msg.get_type() == 'ATTITUDE':
        #         self.pose.roll = msg.roll
        #         self.pose.pitch = msg.pitch
        #         self.pose.yaw = msg.yaw
        #     elif msg.get_type() == 'GLOBAL_POSITION_INT':
        #         self.pose.z = msg.relative_alt / 1000.0  # mm to meters
        #         self.pose.vx = msg.vx / 100.0           # cm/s to m/s
        #         self.pose.vy = msg.vy / 100.0
        #     elif msg.get_type() == 'SYS_STATUS':
        #         self.battery = float(msg.battery_remaining)
        #     msg = self.vehicle.recv_match(type=['ATTITUDE', 'GLOBAL_POSITION_INT', 'SYS_STATUS'], blocking=False)

        return True

    def get_frame(self) -> np.ndarray:
        """On physical drone, frames are acquired from USB Camera (e.g. cv2.VideoCapture(0))."""
        return self.current_frame

    def get_pose(self) -> DronePose:
        return self.pose

    def send_command(self, cmd: MotorCommands) -> None:
        """Sends velocity / position setpoints to Pixhawk in GUIDED mode via MAVLink:

        self.vehicle.mav.set_position_target_local_ned_send(
            0, # time_boot_ms
            self.vehicle.target_system, self.vehicle.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111000111, # type_mask (only speeds enabled)
            0, 0, 0,            # x, y, z positions
            vx, vy, vz,         # velocities in m/s
            0, 0, 0,            # accelerations
            yaw, yaw_rate       # yaw and yaw_rate
        )
        """
        pass

    def get_battery(self) -> float:
        return self.battery

    def get_ground_truth_pedestrians(self) -> List[Dict[str, Any]]:
        return []

    def disconnect(self) -> None:
        if self.vehicle is not None:
            self.vehicle.close()
        self.is_connected = False
