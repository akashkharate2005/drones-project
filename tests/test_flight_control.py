"""Unit tests for DroneFlightController, PID logic, and MissionManager."""
import math
import pytest
from src.utils.config import load_config
from src.control.flight import DroneFlightController, PIDController, DronePose, MotorCommands
from src.control.mission import MissionManager
from src.decision.engine import DecisionOutput, AlertLevel, MissionState


@pytest.fixture
def config():
    return load_config("configs/config.yaml")


def test_pid_controller_basic():
    pid = PIDController(kp=2.0, ki=0.1, kd=0.5, out_min=-10.0, out_max=10.0)
    out = pid.compute(setpoint=10.0, current=0.0, dt=0.1)
    assert -10.0 <= out <= 10.0
    assert out > 0.0 # Error is positive, output should be positive

    # Test clamping
    pid_large = PIDController(kp=100.0, ki=0.0, kd=0.0, out_min=-5.0, out_max=5.0)
    out_clamp = pid_large.compute(setpoint=10.0, current=0.0, dt=0.1)
    assert out_clamp == 5.0


def test_geofence_enforcement(config):
    fc = DroneFlightController(config)
    # Attempt to set waypoint far outside the geofence
    fc.set_target_position(x=100.0, y=-200.0, z=80.0)
    assert fc.target_x == fc.geo_x_max
    assert fc.target_y == fc.geo_y_min
    assert fc.target_z == fc.geo_z_max


def test_motor_mixer_computation(config):
    fc = DroneFlightController(config)
    current_pose = DronePose(x=0.0, y=0.0, z=28.0, roll=0.0, pitch=0.0, yaw=0.0)
    fc.set_target_position(0.0, 0.0, 28.0)

    motors = fc.compute_motor_commands(current_pose, dt=0.032)
    assert isinstance(motors, MotorCommands)
    # Hover balance check
    assert abs(motors.m1_front_left) > 50.0
    assert abs(motors.m2_front_right) > 50.0
    assert motors.gimbal_pitch == pytest.approx(-1.57, abs=0.1)


def test_mission_manager_patrol_progression(config):
    fc = DroneFlightController(config)
    mm = MissionManager(config, fc)

    # Place drone directly at waypoint 0
    wp0 = mm.patrol_waypoints[0]
    pose_at_wp0 = DronePose(x=wp0.x, y=wp0.y, z=wp0.z)

    dec_out = DecisionOutput(
        alert_level=AlertLevel.NORMAL,
        mission_state=MissionState.PATROL,
        trigger_reason="Patrol"
    )

    mm.update(dec_out, pose_at_wp0, dt=0.05)
    # Waypoint should advance to next
    assert mm.current_wp_idx == 1
