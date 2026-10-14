"""Generate high-resolution visual showcase assets for documentation and presentation."""
import os
import sys
import glob
import cv2
import numpy as np

# Ensure root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.perception.detector import PersonDetector
from src.perception.faces import FaceRecognizer
from src.decision.engine import DecisionEngine, AlertLevel, MissionState
from src.alerts.virtual_arduino import VirtualArduino
from src.control.flight import DroneFlightController, DronePose
from src.control.mission import MissionManager
from src.ui.dashboard import Dashboard
from src.sim.video_backend import VideoFileBackend
from face_demo import draw_results, fit_to_tile, add_image_title


def generate_face_grid(output_path="docs/assets/face_demo_grid.jpg"):
    print("[Showcase] Generating face recognition grid asset...")
    config = load_config("configs/config.yaml")
    config["perception"]["faces"]["authorized_dir"] = "data/images"
    config["perception"]["faces"]["authorized_names"] = ["Akash", "Laksh", "Ritu"]
    config["perception"]["faces"]["unknown_dir"] = "data/images/__no_unknown_gallery__"

    recognizer = FaceRecognizer(config)
    test_dir = "data/images/test"
    image_paths = sorted(
        path for pattern in ("*.jpg", "*.jpeg", "*.png")
        for path in glob.glob(os.path.join(test_dir, pattern))
    )

    if not image_paths:
        print("[Showcase] Warning: No test images found.")
        return

    tiles = []
    tile_width, tile_height = 620, 400
    for index, image_path in enumerate(image_paths, start=1):
        frame = cv2.imread(image_path)
        if frame is None:
            continue
        results = recognizer.recognize_faces(frame)
        annotated = draw_results(frame, results, recognizer.backend)
        tile = fit_to_tile(annotated, tile_width, tile_height)
        tiles.append(add_image_title(tile, image_path, index, len(image_paths)))

    columns = min(2, len(tiles))
    rows = (len(tiles) + columns - 1) // columns
    grid = np.full((rows * tile_height, columns * tile_width, 3), (15, 17, 20), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        grid[row * tile_height:(row + 1) * tile_height,
             column * tile_width:(column + 1) * tile_width] = tile

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, grid)
    print(f"[Showcase] Saved face grid to {output_path}")


def generate_hud_snapshots(output_dir="docs/assets"):
    print("[Showcase] Simulating surveillance mission to capture HUD snapshots...")
    os.makedirs(output_dir, exist_ok=True)

    config = load_config("configs/config.yaml")
    backend = VideoFileBackend(config, video_path="")
    backend.connect()

    detector = PersonDetector(config)
    face_recognizer = FaceRecognizer(config)
    decision_engine = DecisionEngine(config)
    arduino = VirtualArduino(config)
    flight_controller = DroneFlightController(config)
    mission_manager = MissionManager(config, flight_controller)
    dashboard = Dashboard(config)

    # Warm up YOLO model
    _ = detector.detect_and_track(backend.get_frame())
    detector.reset_tracker()

    dt = 0.05
    captured = {
        "patrol": False,
        "hotspot": False,
        "inspection": False,
        "rth": False
    }

    sim_time = 0.0
    frame_idx = 0
    inspection_triggered = False

    # Run for up to 65 sim seconds
    while sim_time <= 65.0 and not all(captured.values()):
        sim_time += dt
        frame_idx += 1

        # Phase triggers identical to main.py
        if 15.0 <= sim_time < 35.0:
            if hasattr(backend, "trigger_hotspot_event"):
                backend.trigger_hotspot_event(True)
        elif 35.0 <= sim_time < 55.0:
            if hasattr(backend, "trigger_hotspot_event"):
                backend.trigger_hotspot_event(False)
            if not inspection_triggered:
                decision_engine.trigger_inspection(duration=10.0)
                inspection_triggered = True
        elif sim_time >= 55.0:
            if hasattr(backend, "battery"):
                backend.battery = max(5.0, 18.0 - (sim_time - 55.0) * 0.5)

        backend.step(dt)
        frame = backend.get_frame()
        pose = backend.get_pose()
        battery = backend.get_battery()

        det_result = detector.detect_and_track(frame)
        face_results = []
        if decision_engine.current_mission == MissionState.INSPECT_FACES and pose.z <= 8.0:
            face_results = face_recognizer.recognize_faces(frame)

        decision_out = decision_engine.evaluate(
            crowd_count=det_result.count,
            is_hotspot=getattr(det_result, "is_hotspot", False),
            hotspot_norm=getattr(det_result, "hotspot_centroid", None),
            face_results=face_results,
            battery=battery,
            drone_pose=pose,
            dt=dt
        )

        arduino.update_alert(decision_out.alert_level.value)
        mission_manager.update(decision_out, pose, dt=dt)
        motor_cmd = flight_controller.compute_motor_commands(pose, dt=dt)
        backend.send_command(motor_cmd)

        if hasattr(backend, "set_flight_targets"):
            backend.set_flight_targets(
                flight_controller.target_x,
                flight_controller.target_y,
                flight_controller.target_z,
                flight_controller.target_yaw
            )

        hud_canvas = dashboard.render(
            frame=frame,
            detection_result=det_result,
            face_results=face_results,
            decision_out=decision_out,
            arduino=arduino,
            drone_pose=pose,
            fps=20.0,
            battery=battery
        )

        # Snapshots at designated phases
        if sim_time >= 8.0 and not captured["patrol"]:
            cv2.imwrite(os.path.join(output_dir, "hud_phase1_patrol.jpg"), hud_canvas)
            captured["patrol"] = True
            print("  [x] Captured Phase 1: Patrol HUD")

        if sim_time >= 22.0 and not captured["hotspot"] and decision_out.alert_level == AlertLevel.CRITICAL:
            cv2.imwrite(os.path.join(output_dir, "hud_phase2_hotspot.jpg"), hud_canvas)
            captured["hotspot"] = True
            print("  [x] Captured Phase 2: Hotspot Alert HUD")

        if sim_time >= 42.0 and not captured["inspection"] and pose.z <= 5.5:
            cv2.imwrite(os.path.join(output_dir, "hud_phase3_inspection.jpg"), hud_canvas)
            captured["inspection"] = True
            print("  [x] Captured Phase 3: Facial Inspection HUD")

        if sim_time >= 58.0 and not captured["rth"] and decision_out.mission_state == MissionState.RETURN_HOME:
            cv2.imwrite(os.path.join(output_dir, "hud_phase4_rth.jpg"), hud_canvas)
            captured["rth"] = True
            print("  [x] Captured Phase 4: Emergency RTH HUD")

    backend.disconnect()
    print("[Showcase] HUD snapshots generation complete.")


if __name__ == "__main__":
    generate_face_grid()
    generate_hud_snapshots()

