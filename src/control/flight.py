"""High-level flight API, PID controllers, and rotor mixer for the Mavic 2 Pro."""
from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple


@dataclass
class DronePose:
    """Represents the 6-DOF state of the drone."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0 # Altitude in meters (upwards)
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0 # Heading in radians
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0


@dataclass
class MotorCommands:
    """Angular velocities (rad/s) for the 4 propellers and camera gimbal pitch (rad)."""
    m1_front_left: float
    m2_front_right: float
    m3_rear_left: float
    m4_rear_right: float
    gimbal_pitch: float = -1.57 # Nadir / downward looking (-pi/2)


class PIDController:
    """Standard discrete PID controller with anti-windup clamping."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        out_min: float = -1.0,
        out_max: float = 1.0,
        integral_limit: float = 0.5
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max
        self.integral_limit = integral_limit

        self.integral = 0.0
        self.prev_error = 0.0
        self.first_run = True

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_run = True

    def compute(self, setpoint: float, current: float, dt: float) -> float:
        if dt <= 0.0:
            dt = 0.032

        error = setpoint - current

        # Derivative
        if self.first_run:
            derivative = 0.0
            self.first_run = False
        else:
            derivative = (error - self.prev_error) / dt

        # Integral with anti-windup clamp
        self.integral += error * dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error = error

        return max(self.out_min, min(self.out_max, output))


class DroneFlightController:
    """High-level flight control interface and 4-axis PID autopilot for Mavic 2 Pro."""

    def __init__(self, config: Dict[str, Any]):
        fl_cfg = config.get("flight", {})
        pid_cfg = fl_cfg.get("pid", {})

        self.patrol_altitude = fl_cfg.get("patrol_altitude", 28.0)
        self.inspect_altitude = fl_cfg.get("inspect_altitude", 4.5)
        self.cruise_speed = fl_cfg.get("cruise_speed", 3.0)
        self.max_speed = fl_cfg.get("max_speed", 4.0)
        self.max_climb_speed = fl_cfg.get("max_climb_speed", 2.0)
        home = fl_cfg.get("home_position", [0.0, 0.0, 28.0])
        self.home_x, self.home_y, self.home_z = home[0], home[1], home[2]

        gf = fl_cfg.get("geofence", {})
        self.geo_x_min = gf.get("x_min", -35.0)
        self.geo_x_max = gf.get("x_max", 35.0)
        self.geo_y_min = gf.get("y_min", -35.0)
        self.geo_y_max = gf.get("y_max", 35.0)
        self.geo_z_min = gf.get("z_min", 1.0)
        self.geo_z_max = gf.get("z_max", 40.0)

        # Baseline hover motor velocity for Mavic 2 Pro in Webots (~68.5 rad/s)
        self.base_hover_thrust = 68.5

        # PIDs
        alt_p = pid_cfg.get("altitude", {})
        self.pid_altitude = PIDController(alt_p.get("kp", 2.2), alt_p.get("ki", 0.08), alt_p.get("kd", 1.1),
                                          out_min=-25.0, out_max=25.0, integral_limit=10.0)

        roll_p = pid_cfg.get("roll", {})
        self.pid_roll = PIDController(roll_p.get("kp", 1.8), roll_p.get("ki", 0.04), roll_p.get("kd", 0.75),
                                      out_min=-15.0, out_max=15.0, integral_limit=5.0)

        pitch_p = pid_cfg.get("pitch", {})
        self.pid_pitch = PIDController(pitch_p.get("kp", 1.8), pitch_p.get("ki", 0.04), pitch_p.get("kd", 0.75),
                                       out_min=-15.0, out_max=15.0, integral_limit=5.0)

        yaw_p = pid_cfg.get("yaw", {})
        self.pid_yaw = PIDController(yaw_p.get("kp", 1.4), yaw_p.get("ki", 0.02), yaw_p.get("kd", 0.45),
                                     out_min=-10.0, out_max=10.0, integral_limit=3.0)

        # Target Setpoints
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = self.patrol_altitude
        self.target_yaw = 0.0

        # Orbit helper
        self.orbit_angle = 0.0

    def set_target_position(self, x: float, y: float, z: float, yaw: Optional[float] = None) -> None:
        """Enforces geofence safety limits and sets setpoints."""
        self.target_x = max(self.geo_x_min, min(self.geo_x_max, x))
        self.target_y = max(self.geo_y_min, min(self.geo_y_max, y))
        self.target_z = max(self.geo_z_min, min(self.geo_z_max, z))
        if yaw is not None:
            self.target_yaw = yaw

    def takeoff(self, target_alt: Optional[float] = None) -> None:
        """Command takeoff to cruise altitude."""
        alt = target_alt if target_alt is not None else self.patrol_altitude
        self.set_target_position(self.target_x, self.target_y, alt)

    def goto(self, x: float, y: float, z: float) -> None:
        """Navigate to specific coordinates."""
        # Calculate yaw pointing toward destination
        dx = x - self.target_x
        dy = y - self.target_y
        yaw = math.atan2(dy, dx) if math.hypot(dx, dy) > 1.0 else self.target_yaw
        self.set_target_position(x, y, z, yaw)

    def hover(self) -> None:
        """Hold current horizontal coordinates."""
        pass # Maintains current target setpoints

    def descend_to(self, altitude: float) -> None:
        """Descend safely to target altitude for close inspection."""
        target_z = max(self.geo_z_min, min(self.geo_z_max, altitude))
        self.set_target_position(self.target_x, self.target_y, target_z)

    def return_home(self) -> None:
        """Navigate back to home origin at patrol altitude."""
        self.goto(self.home_x, self.home_y, self.home_z)

    def land(self) -> None:
        """Descend to ground."""
        self.set_target_position(self.target_x, self.target_y, 0.2)

    def orbit(self, center_x: float, center_y: float, radius: float = 6.0, dt: float = 0.05, speed: float = 0.4) -> None:
        """Compute circular orbit path around target coordinates (e.g. crowd hotspot)."""
        self.orbit_angle += speed * dt
        tx = center_x + radius * math.cos(self.orbit_angle)
        ty = center_y + radius * math.sin(self.orbit_angle)
        # Point camera yaw directly toward the center
        yaw = math.atan2(center_y - ty, center_x - tx)
        self.set_target_position(tx, ty, self.patrol_altitude, yaw)

    def compute_motor_commands(self, current_pose: DronePose, dt: float = 0.032) -> MotorCommands:
        """Compute closed-loop motor outputs and camera gimbal pitch based on current pose.

        Args:
            current_pose: Measured drone pose from IMU, GPS, and Gyro.
            dt: Time delta in seconds.

        Returns:
            MotorCommands containing 4 rotor velocities and gimbal pitch angle.
        """
        # 1. Altitude Control
        thrust_adj = self.pid_altitude.compute(self.target_z, current_pose.z, dt)
        total_thrust = self.base_hover_thrust + thrust_adj

        # 2. Horizontal Position to Tilt Mapping
        # In world frame, error vector:
        err_x = self.target_x - current_pose.x
        err_y = self.target_y - current_pose.y

        # Transform world position errors to drone body frame using current yaw
        cos_y = math.cos(current_pose.yaw)
        sin_y = math.sin(current_pose.yaw)
        body_err_fwd = err_x * cos_y + err_y * sin_y
        body_err_lat = -err_x * sin_y + err_y * cos_y

        # Clamp max tilt demand (rad)
        max_tilt = math.radians(16.0)
        target_pitch = max(-max_tilt, min(max_tilt, body_err_fwd * 0.12))
        target_roll = max(-max_tilt, min(max_tilt, -body_err_lat * 0.12))

        # 3. Attitude PIDs
        roll_cmd = self.pid_roll.compute(target_roll, current_pose.roll, dt)
        pitch_cmd = self.pid_pitch.compute(target_pitch, current_pose.pitch, dt)

        # 4. Heading / Yaw PID (wrap error to [-pi, pi])
        yaw_err = (self.target_yaw - current_pose.yaw + math.pi) % (2 * math.pi) - math.pi
        yaw_cmd = self.pid_yaw.compute(yaw_err, 0.0, dt)

        # 5. Rotor Mixing for Quadrotor X-Configuration (Mavic 2 Pro)
        # Propeller rotation signs:
        # m1: front left (CW)
        # m2: front right (CCW)
        # m3: rear left (CCW)
        # m4: rear right (CW)
        m1 = total_thrust - pitch_cmd + roll_cmd - yaw_cmd
        m2 = -(total_thrust - pitch_cmd - roll_cmd + yaw_cmd) # CCW in Webots convention
        m3 = -(total_thrust + pitch_cmd + roll_cmd + yaw_cmd) # CCW
        m4 = total_thrust + pitch_cmd - roll_cmd - yaw_cmd

        # 6. Gimbal Pitch Control
        # Keep camera pointed straight down (-pi/2) relative to ground, compensating drone pitch
        gimbal_pitch = -1.57 - current_pose.pitch

        return MotorCommands(
            m1_front_left=float(m1),
            m2_front_right=float(m2),
            m3_rear_left=float(m3),
            m4_rear_right=float(m4),
            gimbal_pitch=float(gimbal_pitch)
        )
