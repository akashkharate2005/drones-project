"""Autonomous mission planner executing patrol, hotspot orbit, face inspection, and RTH."""
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple

from src.decision.engine import MissionState, DecisionOutput
from src.control.flight import DroneFlightController, DronePose


@dataclass
class Waypoint:
    """3D Waypoint for autonomous navigation."""
    x: float
    y: float
    z: float
    action: Optional[str] = None


class MissionManager:
    """Executes high-level surveillance behaviors dictated by the DecisionEngine."""

    def __init__(self, config: Dict[str, Any], flight_controller: DroneFlightController):
        self.config = config
        self.fc = flight_controller

        fl_cfg = config.get("flight", {})
        self.patrol_altitude = fl_cfg.get("patrol_altitude", 28.0)
        self.inspect_altitude = fl_cfg.get("inspect_altitude", 4.5)

        # Generate Lawnmower sweep pattern across plaza (-20m to +20m)
        self.patrol_waypoints: List[Waypoint] = [
            Waypoint(-18.0, -18.0, self.patrol_altitude),
            Waypoint(18.0, -18.0, self.patrol_altitude),
            Waypoint(18.0, 0.0, self.patrol_altitude),
            Waypoint(-18.0, 0.0, self.patrol_altitude),
            Waypoint(-18.0, 18.0, self.patrol_altitude),
            Waypoint(18.0, 18.0, self.patrol_altitude),
            Waypoint(0.0, 0.0, self.patrol_altitude),
        ]
        self.current_wp_idx = 0

        # Known billboard coordinates for facial inspection checks
        self.inspection_targets: List[Tuple[float, float, str]] = [
            (6.0, 8.0, "Alice"),
            (-10.0, -6.0, "Bob"),
            (12.0, -14.0, "Charlie")
        ]
        self.current_inspect_target: Optional[Tuple[float, float, str]] = None

    def update(
        self,
        decision_out: DecisionOutput,
        current_pose: DronePose,
        dt: float = 0.05
    ) -> None:
        """Translates current mission state from DecisionEngine into flight commands."""
        state = decision_out.mission_state

        if state == MissionState.TAKEOFF:
            self.fc.takeoff(self.patrol_altitude)

        elif state == MissionState.PATROL:
            wp = self.patrol_waypoints[self.current_wp_idx]
            self.fc.goto(wp.x, wp.y, wp.z)

            # Check waypoint arrival (within 2.5 meters)
            dist = math.hypot(wp.x - current_pose.x, wp.y - current_pose.y)
            if dist < 2.5:
                self.current_wp_idx = (self.current_wp_idx + 1) % len(self.patrol_waypoints)

        elif state == MissionState.CROWD_HOTSPOT:
            # Orbit around the detected high-density cluster
            if decision_out.target_hotspot is not None:
                hx, hy = decision_out.target_hotspot
            else:
                hx, hy = current_pose.x, current_pose.y

            self.fc.orbit(center_x=hx, center_y=hy, radius=6.0, dt=dt, speed=0.45)

        elif state == MissionState.INSPECT_FACES:
            # Pick nearest billboard target if not already set
            if self.current_inspect_target is None:
                best_t = self.inspection_targets[0]
                best_d = 999.0
                for t in self.inspection_targets:
                    d = math.hypot(t[0] - current_pose.x, t[1] - current_pose.y)
                    if d < best_d:
                        best_d = d
                        best_t = t
                self.current_inspect_target = best_t

            tx, ty, _ = self.current_inspect_target
            # Fly over target and descend to inspection altitude (4.5m)
            self.fc.set_target_position(tx, ty, self.inspect_altitude)

        elif state == MissionState.UNKNOWN_PERSON:
            # Hover directly in place above the flagged person / location
            self.fc.hover()

        elif state == MissionState.RETURN_HOME:
            self.fc.return_home()
            # If reached home horizontal radius (< 3m), descend and land
            dist_home = math.hypot(self.fc.home_x - current_pose.x, self.fc.home_y - current_pose.y)
            if dist_home < 3.0:
                self.fc.land()

        elif state == MissionState.LANDED:
            self.fc.land()

        # Reset inspection target if we left inspection mode
        if state != MissionState.INSPECT_FACES:
            self.current_inspect_target = None
