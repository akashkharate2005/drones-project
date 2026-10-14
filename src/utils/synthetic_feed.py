"""Synthetic crowd plaza surveillance frame generator and asset creator."""
import math
import os
import random
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np


class SyntheticFeedGenerator:
    """Generates realistic aerial surveillance video frames, pedestrians,

    billboards, and portraits for standalone testing and evaluation.
    """

    def __init__(self, width: int = 960, height: int = 720, num_pedestrians: int = 40):
        self.width = width
        self.height = height
        self.num_pedestrians = num_pedestrians

        # Generate persistent pedestrians with trajectory speeds & clothing
        self.pedestrians = []
        random.seed(42)
        for i in range(num_pedestrians):
            # World coordinates in meters (-30 to +30)
            x = random.uniform(-25.0, 25.0)
            y = random.uniform(-25.0, 25.0)
            vx = random.uniform(-0.8, 0.8)
            vy = random.uniform(-0.8, 0.8)
            color = (random.randint(40, 220), random.randint(40, 220), random.randint(40, 220))
            self.pedestrians.append({
                "id": i + 1,
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "color": color,
                "radius": random.uniform(0.35, 0.45) # meters
            })

        # Billboards placed at fixed coordinates
        self.billboards = [
            {"id": "authorized_alice", "x": 6.0, "y": 8.0, "name": "Alice (Authorized)", "type": "authorized"},
            {"id": "authorized_bob", "x": -10.0, "y": -6.0, "name": "Bob (Authorized)", "type": "authorized"},
            {"id": "unknown_charlie", "x": 12.0, "y": -14.0, "name": "Suspect Charlie", "type": "unknown"}
        ]

    def update_pedestrians(self, dt: float = 0.05, cluster_mode: bool = False) -> None:
        """Update pedestrian positions over time, optionally forming a hotspot cluster."""
        for p in self.pedestrians:
            if cluster_mode and p["id"] <= 28:
                # Direct towards cluster hotspot (x=5.0, y=5.0)
                tx, ty = 5.0, 5.0
                dx = tx - p["x"]
                dy = ty - p["y"]
                dist = math.hypot(dx, dy)
                if dist > 1.0:
                    p["x"] += (dx / dist) * 1.2 * dt
                    p["y"] += (dy / dist) * 1.2 * dt
                else:
                    p["x"] += random.uniform(-0.2, 0.2) * dt
                    p["y"] += random.uniform(-0.2, 0.2) * dt
            else:
                p["x"] += p["vx"] * dt
                p["y"] += p["vy"] * dt

                # Bounce off plaza boundary
                if p["x"] < -26.0 or p["x"] > 26.0:
                    p["vx"] *= -1
                if p["y"] < -26.0 or p["y"] > 26.0:
                    p["vy"] *= -1

    def generate_frame(
        self,
        drone_x: float,
        drone_y: float,
        drone_z: float,
        drone_yaw: float = 0.0,
        gimbal_pitch: float = -1.57 # pointing straight down (-pi/2)
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Render a synthetic top-down camera view from the drone's position and altitude.

        Returns:
            frame: BGR numpy image.
            visible_ground_truth: List of pedestrian coordinates visible in camera FOV.
        """
        frame = np.full((self.height, self.width, 3), (180, 180, 175), dtype=np.uint8)

        # Draw cobblestone/tile grid on ground
        tile_size_px = max(10, int(150.0 / max(5.0, drone_z)))
        for gx in range(0, self.width, tile_size_px):
            cv2.line(frame, (gx, 0), (gx, self.height), (165, 165, 160), 1)
        for gy in range(0, self.height, tile_size_px):
            cv2.line(frame, (0, gy), (self.width, gy), (165, 165, 160), 1)

        # Draw plaza central fountain/monument circle
        fountain_rel_x = 0.0 - drone_x
        fountain_rel_y = 0.0 - drone_y
        f_px = int(self.width / 2 + (fountain_rel_x / drone_z) * (self.width * 0.7))
        f_py = int(self.height / 2 - (fountain_rel_y / drone_z) * (self.height * 0.7))
        f_rad = max(4, int((3.0 / drone_z) * (self.width * 0.5)))
        if -50 <= f_px <= self.width + 50 and -50 <= f_py <= self.height + 50:
            cv2.circle(frame, (f_px, f_py), f_rad, (190, 140, 90), -1)
            cv2.circle(frame, (f_px, f_py), max(2, f_rad - 4), (210, 180, 120), -1)
            cv2.putText(frame, "PLAZA MONUMENT", (f_px - 45, f_py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 50, 50), 1)

        # Scale factor: pixels per meter on ground based on altitude drone_z
        # Mavic 2 Pro FOV ~ 83 deg -> at 30m, ground visible is ~ 50m x 38m
        ppm_x = (self.width * 0.65) / max(2.0, drone_z)
        ppm_y = (self.height * 0.65) / max(2.0, drone_z)

        # Draw billboards
        for b in self.billboards:
            rel_x = b["x"] - drone_x
            rel_y = b["y"] - drone_y
            u = int(self.width / 2 + rel_x * ppm_x)
            v = int(self.height / 2 - rel_y * ppm_y)
            if 0 <= u < self.width and 0 <= v < self.height:
                bw = max(16, int(2.5 * ppm_x))
                bh = max(16, int(2.5 * ppm_y))
                x1, y1 = max(0, u - bw // 2), max(0, v - bh // 2)
                x2, y2 = min(self.width, u + bw // 2), min(self.height, v + bh // 2)
                color = (40, 180, 40) if b["type"] == "authorized" else (40, 40, 220)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
                # When drone descends close (< 8m), draw portrait facial features
                if drone_z < 8.0:
                    face_r = max(5, int(0.6 * ppm_x))
                    cv2.circle(frame, (u, v - 2), face_r, (210, 190, 180), -1)
                    # Eyes
                    cv2.circle(frame, (u - 3, v - 4), 1, (20, 20, 20), -1)
                    cv2.circle(frame, (u + 3, v - 4), 1, (20, 20, 20), -1)
                    # Mouth
                    cv2.line(frame, (u - 3, v + 2), (u + 3, v + 2), (20, 20, 120), 1)
                    cv2.putText(frame, b["name"][:8], (x1, y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # Draw pedestrians and collect visible ground truth
        visible_pedestrians = []
        for p in self.pedestrians:
            rel_x = p["x"] - drone_x
            rel_y = p["y"] - drone_y

            # Perspective projection to screen pixel coordinates
            u = int(self.width / 2 + rel_x * ppm_x)
            v = int(self.height / 2 - rel_y * ppm_y)

            # Check if within frame bounds with slight margin
            if 15 <= u < self.width - 15 and 15 <= v < self.height - 15:
                visible_pedestrians.append({
                    "id": p["id"],
                    "x": p["x"],
                    "y": p["y"],
                    "u": u,
                    "v": v
                })

                # Size of person in pixels shrinks with altitude
                # Person shoulder width ~ 0.5m, height from above ~ 0.7m ellipse
                rx = max(3, int(p["radius"] * ppm_x))
                ry = max(4, int(p["radius"] * 1.3 * ppm_y))

                # Drop shadow
                cv2.ellipse(frame, (u + 2, v + 2), (rx, ry), 0, 0, 360, (130, 130, 130), -1)

                # Torso / clothing
                cv2.ellipse(frame, (u, v), (rx, ry), 0, 0, 360, p["color"], -1)

                # Head (viewed from above)
                head_r = max(2, int(rx * 0.55))
                cv2.circle(frame, (u, v - int(ry * 0.2)), head_r, (215, 195, 175), -1)
                # Hair
                hair_color = (30, 20, 20) if p["id"] % 2 == 0 else (60, 45, 30)
                cv2.circle(frame, (u, v - int(ry * 0.25)), max(1, head_r - 1), hair_color, -1)

        return frame, visible_pedestrians


def generate_sample_assets(output_dir: str = "data") -> None:
    """Generate authorized portrait photos, unknown suspect photo, and billboard textures."""
    auth_dir = os.path.join(output_dir, "authorized")
    unk_dir = os.path.join(output_dir, "unknown")
    bb_dir = os.path.join(output_dir, "billboards")
    samples_dir = os.path.join(output_dir, "samples")

    os.makedirs(os.path.join(auth_dir, "alice"), exist_ok=True)
    os.makedirs(os.path.join(auth_dir, "bob"), exist_ok=True)
    os.makedirs(os.path.join(unk_dir, "suspect_charlie"), exist_ok=True)
    os.makedirs(bb_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    def draw_portrait(name: str, bg_color: Tuple[int, int, int], is_authorized: bool) -> np.ndarray:
        """Create a clear 320x320 portrait with realistic facial geometry."""
        img = np.full((320, 320, 3), bg_color, dtype=np.uint8)
        # Shoulders
        cv2.ellipse(img, (160, 310), (100, 70), 0, 0, 360, (60, 60, 80), -1)
        # Neck
        cv2.rectangle(img, (140, 210), (180, 260), (200, 175, 150), -1)
        # Face contour
        cv2.ellipse(img, (160, 160), (60, 80), 0, 0, 360, (220, 190, 165), -1)
        # Hair
        cv2.ellipse(img, (160, 115), (65, 45), 0, 180, 360, (30, 25, 20), -1)
        # Eyes
        cv2.circle(img, (135, 145), 7, (255, 255, 255), -1)
        cv2.circle(img, (185, 145), 7, (255, 255, 255), -1)
        cv2.circle(img, (135, 145), 3, (20, 20, 40), -1)
        cv2.circle(img, (185, 145), 3, (20, 20, 40), -1)
        # Eyebrows
        cv2.line(img, (125, 133), (145, 135), (30, 20, 10), 2)
        cv2.line(img, (175, 135), (195, 133), (30, 20, 10), 2)
        # Nose
        cv2.line(img, (160, 150), (158, 175), (180, 150, 130), 2)
        cv2.line(img, (158, 175), (165, 175), (180, 150, 130), 2)
        # Mouth
        cv2.line(img, (145, 195), (175, 195), (150, 70, 70), 2)
        # Header banner
        status_text = "AUTHORIZED" if is_authorized else "UNKNOWN / SUSPECT"
        status_color = (0, 160, 0) if is_authorized else (0, 0, 200)
        cv2.rectangle(img, (0, 0), (320, 40), status_color, -1)
        cv2.putText(img, status_text, (20, 26), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        cv2.rectangle(img, (0, 280), (320, 320), (30, 30, 30), -1)
        cv2.putText(img, name, (20, 308), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        return img

    alice_img = draw_portrait("Alice Johnson", (220, 230, 220), True)
    bob_img = draw_portrait("Bob Miller", (230, 225, 220), True)
    charlie_img = draw_portrait("Charlie Vance", (210, 210, 225), False)

    cv2.imwrite(os.path.join(auth_dir, "alice", "alice_1.jpg"), alice_img)
    cv2.imwrite(os.path.join(auth_dir, "bob", "bob_1.jpg"), bob_img)
    cv2.imwrite(os.path.join(unk_dir, "suspect_charlie", "suspect_1.jpg"), charlie_img)

    # Billboard textures for Webots world
    cv2.imwrite(os.path.join(bb_dir, "billboard_alice.jpg"), alice_img)
    cv2.imwrite(os.path.join(bb_dir, "billboard_bob.jpg"), bob_img)
    cv2.imwrite(os.path.join(bb_dir, "billboard_charlie.jpg"), charlie_img)


def generate_sample_video(output_path: str = "data/samples/crowd_surveillance_demo.mp4", num_frames: int = 150) -> None:
    """Renders a sample 150-frame surveillance video for instant VideoFileBackend testing."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    generator = SyntheticFeedGenerator(width=960, height=720, num_pedestrians=35)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, 20.0, (960, 720))

    # Simulate drone slow sweeping motion
    for i in range(num_frames):
        generator.update_pedestrians(dt=0.05, cluster_mode=(i > 60))
        t = i / 20.0
        drone_x = 4.0 * math.sin(t * 0.4)
        drone_y = 3.0 * math.cos(t * 0.3)
        drone_z = 25.0 - (15.0 * (i / num_frames) if i > 80 else 0.0)

        frame, _ = generator.generate_frame(drone_x, drone_y, drone_z)
        writer.write(frame)

    writer.release()
