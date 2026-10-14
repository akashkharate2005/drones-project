"""Tactical multi-panel OpenCV HUD dashboard for real-time surveillance operations."""
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.decision.engine import AlertLevel, MissionState, DecisionOutput
from src.alerts.virtual_arduino import VirtualArduino, LEDColor, BuzzerMode
from src.control.flight import DronePose


class Dashboard:
    """Renders a rich 1280x720 tactical HUD combining annotated camera video,

    thermal crowd density overlay, 2D radar minimap, virtual Arduino payload, and event logs.
    """

    def __init__(self, config: Dict[str, Any], window_title: str = "Smart Crowd Surveillance Payload - HUD"):
        self.window_title = window_title
        self.canvas_w = 1280
        self.canvas_h = 720

        # Primary camera feed viewport
        self.cam_w = 840
        self.cam_h = 560

        # Minimap viewport
        self.map_w = 400
        self.map_h = 280

        # Trajectory history for radar map
        self.flight_trail: List[Tuple[float, float]] = []
        self.max_trail_points = 120

        # Event log messages: list of (time_str, message, color)
        self.log_messages: List[Tuple[str, str, Tuple[int, int, int]]] = []
        self.add_log("System initialized in autonomous mode", (180, 180, 180))

        self.last_alert_level = AlertLevel.NORMAL

    def add_log(self, text: str, color: Tuple[int, int, int] = (220, 220, 220)) -> None:
        """Add an entry to the HUD event log."""
        t_str = time.strftime("%H:%M:%S")
        self.log_messages.append((t_str, text, color))
        if len(self.log_messages) > 7:
            self.log_messages.pop(0)

    def render(
        self,
        frame: np.ndarray,
        detection_result: Any,
        face_results: List[Any],
        decision_out: DecisionOutput,
        arduino: VirtualArduino,
        drone_pose: DronePose,
        fps: float,
        battery: float,
        show_density_overlay: bool = True
    ) -> np.ndarray:
        """Assembles all telemetry, vision, and payload panels onto the master HUD canvas."""
        canvas = np.full((self.canvas_h, self.canvas_w, 3), (24, 26, 28), dtype=np.uint8)

        # Log state changes
        if decision_out.alert_level != self.last_alert_level:
            c = (40, 220, 40) if decision_out.alert_level == AlertLevel.NORMAL else (
                (30, 180, 240) if decision_out.alert_level == AlertLevel.WARNING else (40, 40, 240)
            )
            self.add_log(f"ALERT: {decision_out.alert_level.value} - {decision_out.trigger_reason}", c)
            self.last_alert_level = decision_out.alert_level

        # Record trail
        self.flight_trail.append((drone_pose.x, drone_pose.y))
        if len(self.flight_trail) > self.max_trail_points:
            self.flight_trail.pop(0)

        # 1. Render Main Annotated Video Panel
        cam_panel = self._render_camera_view(frame, detection_result, face_results, show_density_overlay)
        canvas[50:50 + self.cam_h, 20:20 + self.cam_w] = cam_panel

        # 2. Render Top Header Banner
        self._render_header(canvas, decision_out, fps, battery, drone_pose)

        # 3. Render Tactical Radar Minimap
        self._render_radar_map(canvas, drone_pose, decision_out)

        # 4. Render Virtual Arduino Payload Module
        self._render_virtual_arduino_panel(canvas, arduino)

        # 5. Render Scrolling Event Log Panel
        self._render_event_log(canvas)

        return canvas

    def _render_camera_view(
        self,
        frame: np.ndarray,
        detection_result: Any,
        face_results: List[Any],
        show_density_overlay: bool
    ) -> np.ndarray:
        """Annotate camera feed with YOLO bounding boxes, ByteTrack IDs, faces, and density heatmaps."""
        view = cv2.resize(frame, (self.cam_w, self.cam_h))
        scale_x = self.cam_w / max(1, frame.shape[1])
        scale_y = self.cam_h / max(1, frame.shape[0])

        # Draw Density Heatmap Overlay
        if show_density_overlay and hasattr(detection_result, "density_map"):
            dmap = detection_result.density_map
            if dmap is not None and np.max(dmap) > 0.1:
                norm_dmap = cv2.normalize(dmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                color_map = cv2.applyColorMap(norm_dmap, cv2.COLORMAP_JET)
                resized_map = cv2.resize(color_map, (self.cam_w, self.cam_h), interpolation=cv2.INTER_LINEAR)
                # Alpha blend
                view = cv2.addWeighted(view, 0.72, resized_map, 0.28, 0)

        # Draw Person BBoxes & Track IDs
        if hasattr(detection_result, "boxes"):
            for i, (x1, y1, x2, y2) in enumerate(detection_result.boxes):
                sx1, sy1 = int(x1 * scale_x), int(y1 * scale_y)
                sx2, sy2 = int(x2 * scale_x), int(y2 * scale_y)
                tid = detection_result.track_ids[i] if i < len(detection_result.track_ids) else (i + 1)

                cv2.rectangle(view, (sx1, sy1), (sx2, sy2), (0, 220, 220), 2)
                tag = f"ID:{tid}"
                cv2.putText(view, tag, (sx1, max(14, sy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 240, 240), 1)

        # Draw Face Match BBoxes
        for fm in face_results:
            fx1, fy1, fx2, fy2 = fm.box
            sfx1, sfy1 = int(fx1 * scale_x), int(fy1 * scale_y)
            sfx2, sfy2 = int(fx2 * scale_x), int(fy2 * scale_y)

            color = (40, 220, 40) if fm.is_authorized else (40, 40, 240)
            cv2.rectangle(view, (sfx1, sfy1), (sfx2, sfy2), color, 2)
            label = f"{fm.name} ({fm.confidence:.2f})"
            cv2.putText(view, label, (sfx1, max(16, sfy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

        # Hotspot target indicator
        if getattr(detection_result, "is_hotspot", False) and detection_result.hotspot_centroid:
            nx, ny = detection_result.hotspot_centroid
            hx, hy = int(nx * self.cam_w), int(ny * self.cam_h)
            cv2.drawMarker(view, (hx, hy), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 24, 2)
            cv2.putText(view, "HOTSPOT", (hx + 14, hy - 6), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 255), 1)

        # Camera viewport border
        cv2.rectangle(view, (0, 0), (self.cam_w - 1, self.cam_h - 1), (80, 80, 80), 1)

        # Live stats pill inside camera view
        cnt = getattr(detection_result, "count", 0)
        cv2.rectangle(view, (10, 10), (220, 42), (20, 20, 20), -1)
        cv2.putText(view, f"PERSONS: {cnt}  TRACKS: {getattr(detection_result, 'unique_tracked_count', 0)}",
                    (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 200), 1)

        return view

    def _render_header(
        self,
        canvas: np.ndarray,
        decision_out: DecisionOutput,
        fps: float,
        battery: float,
        pose: DronePose
    ) -> None:
        """Top telemetry and alert header strip."""
        cv2.rectangle(canvas, (0, 0), (self.canvas_w, 44), (16, 18, 20), -1)

        # Title
        cv2.putText(canvas, "AI SURVEILLANCE PAYLOAD", (20, 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)

        # Alert Badge
        alert_col = (40, 200, 40)
        if decision_out.alert_level == AlertLevel.WARNING:
            alert_col = (30, 160, 240)
        elif decision_out.alert_level == AlertLevel.CRITICAL:
            alert_col = (40, 40, 240)

        cv2.rectangle(canvas, (320, 8), (470, 36), alert_col, -1)
        cv2.putText(canvas, f"ALERT: {decision_out.alert_level.value}", (330, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

        # Mission State
        cv2.rectangle(canvas, (485, 8), (665, 36), (50, 50, 60), -1)
        cv2.putText(canvas, f"MODE: {decision_out.mission_state.value}", (495, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

        # Telemetry: Alt, Speed, Battery, FPS
        spd = math.hypot(pose.vx, pose.vy)
        t_info = f"ALT: {pose.z:.1f}m   SPD: {spd:.1f}m/s   FPS: {fps:.1f}"
        cv2.putText(canvas, t_info, (685, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

        # Battery Bar
        bat_col = (40, 200, 40) if battery > 40 else ((30, 160, 240) if battery > 20 else (40, 40, 240))
        cv2.rectangle(canvas, (1130, 12), (1250, 32), (60, 60, 60), 1)
        fill_w = int((battery / 100.0) * 118)
        if fill_w > 0:
            cv2.rectangle(canvas, (1131, 13), (1131 + fill_w, 31), bat_col, -1)
        cv2.putText(canvas, f"{battery:.0f}%", (1170, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    def _render_radar_map(self, canvas: np.ndarray, pose: DronePose, decision_out: DecisionOutput) -> None:
        """Top-down 2D radar minimap of the plaza."""
        mx, my = 880, 50
        map_img = np.full((self.map_h, self.map_w, 3), (32, 34, 38), dtype=np.uint8)

        # Draw grid
        cx, cy = self.map_w // 2, self.map_h // 2
        cv2.line(map_img, (0, cy), (self.map_w, cy), (48, 52, 58), 1)
        cv2.line(map_img, (cx, 0), (cx, self.map_h), (48, 52, 58), 1)
        cv2.circle(map_img, (cx, cy), 60, (48, 52, 58), 1)
        cv2.circle(map_img, (cx, cy), 120, (48, 52, 58), 1)

        # Scale factor: 30m maps to 140 pixels
        scale = 140.0 / 30.0

        # Draw Billboards on radar
        bb_coords = [(6.0, 8.0, (40, 200, 40)), (-10.0, -6.0, (40, 200, 40)), (12.0, -14.0, (40, 40, 240))]
        for bx, by, bcol in bb_coords:
            px = int(cx + bx * scale)
            py = int(cy - by * scale)
            if 0 <= px < self.map_w and 0 <= py < self.map_h:
                cv2.rectangle(map_img, (px - 3, py - 3), (px + 3, py + 3), bcol, -1)

        # Draw Flight Trail
        for tx, ty in self.flight_trail:
            px = int(cx + tx * scale)
            py = int(cy - ty * scale)
            if 0 <= px < self.map_w and 0 <= py < self.map_h:
                cv2.circle(map_img, (px, py), 1, (0, 180, 255), -1)

        # Draw Drone Position & Heading
        dx = int(cx + pose.x * scale)
        dy = int(cy - pose.y * scale)
        if 0 <= dx < self.map_w and 0 <= dy < self.map_h:
            cv2.circle(map_img, (dx, dy), 6, (0, 240, 255), -1)
            # Heading vector
            hx = int(dx + 14 * math.cos(pose.yaw))
            hy = int(dy - 14 * math.sin(pose.yaw))
            cv2.arrowedLine(map_img, (dx, dy), (hx, hy), (255, 255, 255), 2, tipLength=0.35)

        # Title
        cv2.putText(map_img, "TACTICAL RADAR (60x60m PLAZA)", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
        cv2.rectangle(map_img, (0, 0), (self.map_w - 1, self.map_h - 1), (80, 80, 80), 1)

        canvas[my:my + self.map_h, mx:mx + self.map_w] = map_img

    def _render_virtual_arduino_panel(self, canvas: np.ndarray, arduino: VirtualArduino) -> None:
        """Emulated Arduino hardware payload panel with tri-color LED and buzzer state."""
        ax, ay = 880, 340
        aw, ah = self.map_w, 140
        ard_img = np.full((ah, aw, 3), (28, 30, 34), dtype=np.uint8)

        # Header
        cv2.putText(ard_img, "PAYLOAD: VIRTUAL ARDUINO NANO", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        # Large Glowing Tri-Color LED
        led_color = arduino.get_ui_led_rgb()
        cv2.circle(ard_img, (50, 65), 20, (50, 50, 50), -1) # Socket
        cv2.circle(ard_img, (50, 65), 15, led_color, -1)     # Glow
        cv2.circle(ard_img, (46, 60), 4, (255, 255, 255), -1) # Specular highlight

        cv2.putText(ard_img, f"LED STATUS: {arduino.led_color.value}", (85, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        cv2.putText(ard_img, arduino.get_buzzer_status_text(), (85, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 200, 255), 1)

        # Serial TX Packet Monitor
        pkt = arduino.last_tx_packet or "<STANDBY>\n"
        cv2.rectangle(ard_img, (10, 102), (aw - 10, 128), (14, 16, 18), -1)
        cv2.putText(ard_img, f"UART TX: {pkt.strip()}", (16, 120), cv2.FONT_HERSHEY_PLAIN, 0.85, (0, 220, 140), 1)

        cv2.rectangle(ard_img, (0, 0), (aw - 1, ah - 1), (80, 80, 80), 1)
        canvas[ay:ay + ah, ax:ax + aw] = ard_img

    def _render_event_log(self, canvas: np.ndarray) -> None:
        """Real-time scrolling event log panel."""
        lx, ly = 880, 490
        lw, lh = self.map_w, 220
        log_img = np.full((lh, lw, 3), (20, 22, 24), dtype=np.uint8)

        cv2.putText(log_img, "MISSION EVENT LOG", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
        cv2.line(log_img, (10, 28), (lw - 10, 28), (40, 44, 48), 1)

        y_offset = 50
        for t_str, msg, col in self.log_messages:
            entry = f"[{t_str}] {msg[:36]}"
            cv2.putText(log_img, entry, (12, y_offset), cv2.FONT_HERSHEY_PLAIN, 0.82, col, 1)
            y_offset += 24

        cv2.rectangle(log_img, (0, 0), (lw - 1, lh - 1), (80, 80, 80), 1)
        canvas[ly:ly + lh, lx:lx + lw] = log_img

    def show(self, canvas: np.ndarray, wait_ms: int = 1) -> int:
        """Display image in OpenCV window and return keypress."""
        cv2.imshow(self.window_title, canvas)
        return cv2.waitKey(wait_ms) & 0xFF

    def close(self) -> None:
        cv2.destroyAllWindows()
