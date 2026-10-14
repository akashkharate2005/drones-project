"""Decision engine state machine with hysteresis debouncing and prioritized alert transitions."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class AlertLevel(Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class MissionState(Enum):
    IDLE = "IDLE"
    TAKEOFF = "TAKEOFF"
    PATROL = "PATROL"
    CROWD_HOTSPOT = "CROWD_HOTSPOT"
    INSPECT_FACES = "INSPECT_FACES"
    UNKNOWN_PERSON = "UNKNOWN_PERSON"
    RETURN_HOME = "RETURN_HOME"
    LANDED = "LANDED"


@dataclass
class DecisionOutput:
    """Outputs computed by the decision engine for flight control and alerting."""
    alert_level: AlertLevel
    mission_state: MissionState
    trigger_reason: str
    target_hotspot: Optional[Tuple[float, float]] = None # (world_x, world_y)
    tracked_suspect_id: Optional[int] = None
    state_changed: bool = False
    details: str = ""


class DecisionEngine:
    """Autonomous decision engine evaluating sensor, vision, and battery inputs.

    Incorporates hysteresis to prevent rapid flickering of alert levels and mission states.
    """

    def __init__(self, config: Dict[str, Any]):
        dec_cfg = config.get("decision", {})
        self.crowd_warning_threshold = dec_cfg.get("crowd_warning_threshold", 12)
        self.crowd_critical_threshold = dec_cfg.get("crowd_critical_threshold", 25)
        self.hysteresis_frames = dec_cfg.get("hysteresis_frames", 8)
        self.battery_critical_threshold = dec_cfg.get("battery_critical_threshold", 20.0)
        self.battery_warning_threshold = dec_cfg.get("battery_warning_threshold", 30.0)
        self.inspection_duration_sec = dec_cfg.get("inspection_duration_sec", 6.0)

        # Current stable states
        self.current_alert = AlertLevel.NORMAL
        self.current_mission = MissionState.TAKEOFF

        # Hysteresis tracking
        self.candidate_alert = AlertLevel.NORMAL
        self.candidate_alert_counter = 0

        # Inspection timer
        self.inspection_timer = 0.0
        self.suspect_hold_timer = 0.0
        self.last_unknown_track_id: Optional[int] = None

    def evaluate(
        self,
        crowd_count: int,
        is_hotspot: bool,
        hotspot_norm: Optional[Tuple[float, float]],
        face_results: List[Any],
        battery: float,
        drone_pose: Any,
        dt: float = 0.05
    ) -> DecisionOutput:
        """Step the decision engine with new frame perception and telemetry.

        Returns:
            DecisionOutput with stabilized alert level and target mission state.
        """
        prev_alert = self.current_alert
        prev_mission = self.current_mission

        # 1. Check for Unknown / Flagged Suspect faces
        has_unknown_face = False
        unknown_desc = ""
        for fm in face_results:
            name = getattr(fm, "name", "")
            is_auth = getattr(fm, "is_authorized", False)
            if not is_auth or "SUSPECT" in name:
                has_unknown_face = True
                unknown_desc = name if name != "Unknown" else "Unidentified Individual"
                break

        # 2. Determine raw alert candidate based on priority
        raw_alert = AlertLevel.NORMAL
        raw_reason = "Nominal conditions"

        if battery <= self.battery_critical_threshold:
            raw_alert = AlertLevel.CRITICAL
            raw_reason = f"Critical Battery: {battery:.1f}% <= {self.battery_critical_threshold}%"
        elif has_unknown_face:
            raw_alert = AlertLevel.CRITICAL
            raw_reason = f"Security Breach: {unknown_desc} Detected"
        elif crowd_count >= self.crowd_critical_threshold:
            raw_alert = AlertLevel.CRITICAL
            raw_reason = f"Severe Overcrowding: {crowd_count} >= {self.crowd_critical_threshold}"
        elif crowd_count >= self.crowd_warning_threshold or is_hotspot:
            raw_alert = AlertLevel.WARNING
            raw_reason = (
                f"Crowd Congestion ({crowd_count} persons)"
                if crowd_count >= self.crowd_warning_threshold
                else "Localized Crowd Hotspot Detected"
            )
        elif battery <= self.battery_warning_threshold:
            raw_alert = AlertLevel.WARNING
            raw_reason = f"Low Battery: {battery:.1f}%"

        # 3. Apply Hysteresis
        # Security threats and critical battery escalate instantly without debounce delay
        if raw_alert == AlertLevel.CRITICAL and (has_unknown_face or battery <= self.battery_critical_threshold):
            self.current_alert = AlertLevel.CRITICAL
            self.candidate_alert = AlertLevel.CRITICAL
            self.candidate_alert_counter = 0
        else:
            if raw_alert == self.candidate_alert:
                self.candidate_alert_counter += 1
                if self.candidate_alert_counter >= self.hysteresis_frames:
                    self.current_alert = self.candidate_alert
            else:
                self.candidate_alert = raw_alert
                self.candidate_alert_counter = 1

        # 4. Mission State Transition Logic
        target_mission = self.current_mission

        # Low battery override
        if battery <= self.battery_critical_threshold:
            target_mission = MissionState.RETURN_HOME

        # Security breach tracking
        elif has_unknown_face:
            target_mission = MissionState.UNKNOWN_PERSON
            self.suspect_hold_timer = 5.0 # Keep tracking for 5 seconds

        elif target_mission == MissionState.UNKNOWN_PERSON:
            self.suspect_hold_timer -= dt
            if self.suspect_hold_timer <= 0.0:
                target_mission = MissionState.PATROL

        # Active crowd hotspot
        elif self.current_alert in [AlertLevel.WARNING, AlertLevel.CRITICAL] and crowd_count >= self.crowd_warning_threshold:
            if target_mission != MissionState.INSPECT_FACES:
                target_mission = MissionState.CROWD_HOTSPOT

        # Facial inspection handling
        elif target_mission == MissionState.INSPECT_FACES:
            self.inspection_timer -= dt
            if self.inspection_timer <= 0.0:
                target_mission = MissionState.PATROL

        elif target_mission in [MissionState.TAKEOFF, MissionState.IDLE]:
            # If drone reached cruise altitude, commence patrol
            drone_z = getattr(drone_pose, "z", 0.0)
            if drone_z >= 15.0:
                target_mission = MissionState.PATROL

        elif target_mission == MissionState.CROWD_HOTSPOT and self.current_alert == AlertLevel.NORMAL:
            target_mission = MissionState.PATROL

        self.current_mission = target_mission

        # Compute hotspot ground target if available
        target_hotspot_world = None
        if hotspot_norm is not None and drone_pose is not None:
            # Map normalized image hotspot to ground world coordinates
            norm_x, norm_y = hotspot_norm
            drone_z = max(2.0, getattr(drone_pose, "z", 28.0))
            offset_x = (norm_x - 0.5) * (drone_z * 1.2)
            offset_y = (0.5 - norm_y) * (drone_z * 1.2)
            target_hotspot_world = (
                getattr(drone_pose, "x", 0.0) + offset_x,
                getattr(drone_pose, "y", 0.0) + offset_y
            )

        state_changed = (self.current_alert != prev_alert) or (self.current_mission != prev_mission)

        return DecisionOutput(
            alert_level=self.current_alert,
            mission_state=self.current_mission,
            trigger_reason=raw_reason,
            target_hotspot=target_hotspot_world,
            state_changed=state_changed,
            details=f"Count: {crowd_count}, Battery: {battery:.1f}%"
        )

    def trigger_inspection(self, duration: Optional[float] = None) -> None:
        """Manually or mission-triggered facial inspection descent."""
        self.current_mission = MissionState.INSPECT_FACES
        self.inspection_timer = duration if duration is not None else self.inspection_duration_sec
