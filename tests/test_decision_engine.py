"""Unit tests for the decision engine state machine and hysteresis."""
import pytest
from src.utils.config import load_config
from src.decision.engine import DecisionEngine, AlertLevel, MissionState
from src.perception.faces import FaceMatchResult


class DummyPose:
    def __init__(self, x=0.0, y=0.0, z=28.0, yaw=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw


@pytest.fixture
def engine():
    cfg = load_config("configs/config.yaml")
    return DecisionEngine(cfg)


def test_initial_state(engine):
    assert engine.current_alert == AlertLevel.NORMAL
    assert engine.current_mission == MissionState.TAKEOFF


def test_takeoff_to_patrol_transition(engine):
    pose = DummyPose(z=25.0)
    out = engine.evaluate(crowd_count=5, is_hotspot=False, hotspot_norm=None,
                          face_results=[], battery=95.0, drone_pose=pose)
    assert out.alert_level == AlertLevel.NORMAL
    assert out.mission_state == MissionState.PATROL


def test_hysteresis_debouncing(engine):
    pose = DummyPose(z=25.0)
    # Put into patrol first
    engine.evaluate(crowd_count=5, is_hotspot=False, hotspot_norm=None,
                    face_results=[], battery=95.0, drone_pose=pose)

    # 1 single frame of crowd count = 16 (warning threshold is 12)
    # Should NOT immediately change alert level to WARNING because hysteresis_frames is 8
    out1 = engine.evaluate(crowd_count=16, is_hotspot=False, hotspot_norm=None,
                           face_results=[], battery=95.0, drone_pose=pose)
    assert out1.alert_level == AlertLevel.NORMAL

    # Sustain high count for 7 more frames (total 8 frames)
    for _ in range(7):
        out = engine.evaluate(crowd_count=16, is_hotspot=False, hotspot_norm=None,
                              face_results=[], battery=95.0, drone_pose=pose)

    # Now it should have transitioned to WARNING
    assert out.alert_level == AlertLevel.WARNING
    assert out.mission_state == MissionState.CROWD_HOTSPOT


def test_immediate_escalation_on_unknown_face(engine):
    pose = DummyPose(z=25.0)
    # Face match of unknown suspect
    face_match = FaceMatchResult(
        box=(50, 50, 150, 150),
        name="SUSPECT: Charlie",
        is_authorized=False,
        confidence=0.88,
        backend_used="opencv"
    )

    out = engine.evaluate(crowd_count=4, is_hotspot=False, hotspot_norm=None,
                          face_results=[face_match], battery=80.0, drone_pose=pose)

    # Instant jump without debounce delay
    assert out.alert_level == AlertLevel.CRITICAL
    assert out.mission_state == MissionState.UNKNOWN_PERSON


def test_low_battery_return_home(engine):
    pose = DummyPose(z=25.0)
    out = engine.evaluate(crowd_count=4, is_hotspot=False, hotspot_norm=None,
                          face_results=[], battery=18.0, drone_pose=pose)

    assert out.alert_level == AlertLevel.CRITICAL
    assert out.mission_state == MissionState.RETURN_HOME
