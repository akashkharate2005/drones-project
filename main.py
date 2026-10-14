"""Unified CLI entry point for AI-Based Smart Crowd Monitoring Payload.

Usage:
    python main.py --mode scenario                 # Automated full demo scenario
    python main.py --mode video --video <path>     # Process recorded video
    python main.py --mode webcam                   # Live USB webcam feed
    python main.py --mode webots                   # Connect to Webots EXTERN controller
"""
import argparse
import os
import sys
import time
from typing import Any, Dict, Optional
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils.config import load_config
from src.utils.logger import MissionLogger
from src.perception.detector import PersonDetector
from src.perception.faces import FaceRecognizer
from src.decision.engine import DecisionEngine, AlertLevel, MissionState
from src.alerts.virtual_arduino import VirtualArduino
from src.control.flight import DroneFlightController, DronePose
from src.control.mission import MissionManager
from src.ui.dashboard import Dashboard
from src.sim.base import SimulatorBackend
from src.sim.webots_backend import WebotsBackend
from src.sim.video_backend import VideoFileBackend
from src.sim.webcam_backend import WebcamBackend


def create_backend(mode: str, config: Dict[str, Any], args: argparse.Namespace) -> SimulatorBackend:
    """Instantiate appropriate simulation backend based on selected mode."""
    if mode == "webots":
        return WebotsBackend(config)
    elif mode == "webcam":
        return WebcamBackend(config, camera_index=args.camera_id)
    elif mode == "video":
        v_path = args.video if args.video else "data/samples/crowd_surveillance_demo.mp4"
        return VideoFileBackend(config, video_path=v_path)
    else: # "scenario"
        # Dynamic synthetic renderer perfectly synchronized with 3D flight navigation
        return VideoFileBackend(config, video_path="")


def run_mission(args: argparse.Namespace) -> None:
    # 1. Load Configuration
    config = load_config(args.config)
    target_fps = config.get("system", {}).get("target_fps", 20)
    dt = 1.0 / target_fps
    frame_skip = config.get("system", {}).get("frame_skip", 1)

    print(f"================================================================")
    print(f" AI-Based Smart Crowd Monitoring Payload for Surveillance Drones")
    print(f" Mode: {args.mode.upper()} | Target FPS: {target_fps}")
    print(f"================================================================")

    # 2. Initialize Subsystems
    logger = MissionLogger(config)
    detector = PersonDetector(config)
    face_recognizer = FaceRecognizer(config)
    decision_engine = DecisionEngine(config)
    arduino = VirtualArduino(config)
    flight_controller = DroneFlightController(config)
    mission_manager = MissionManager(config, flight_controller)
    dashboard = Dashboard(config)

    # 3. Initialize Simulator Backend
    backend = create_backend(args.mode, config, args)
    if not backend.connect():
        if args.mode == "webots":
            print("[Warning] Webots connection unavailable. Falling back automatically to VideoFileBackend...")
            backend = VideoFileBackend(config, video_path="data/samples/crowd_surveillance_demo.mp4")
            backend.connect()
        else:
            print("[Error] Failed to initialize backend.")
            return

    # Video recording setup if requested
    video_writer = None
    if args.record:
        os.makedirs(os.path.dirname(args.record), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.record, fourcc, float(target_fps), (1280, 720))
        print(f"[Recorder] Recording HUD feed to {args.record}")

    # Warm up YOLO model
    print("[Mission] Warming up perception detector...")
    _ = detector.detect_and_track(backend.get_frame())
    detector.reset_tracker()
    print("[Mission] Subsystems ready. Commencing autonomous surveillance.")

    frame_idx = 0
    start_time = time.time()
    sim_time = 0.0
    inspection_triggered = False

    # Cached perception outputs for skipped frames
    last_det_result = None
    last_face_results = []

    try:
        while True:
            loop_start = time.time()
            sim_time += dt

            # Max duration check based on simulation time
            if args.duration > 0 and sim_time >= args.duration:
                print(f"[Mission] Reached specified mission duration limit ({args.duration}s). Concluding mission.")
                break

            # -------------------------------------------------------------
            # Scenario Mode Scripted Sequences (Timeline based on sim_time)
            # -------------------------------------------------------------
            if args.mode == "scenario":
                # Phase 1: 0 - 15s -> Normal Patrol
                # Phase 2: 15s - 35s -> Crowd Hotspot Cluster Forms
                if 15.0 <= sim_time < 35.0:
                    if hasattr(backend, "trigger_hotspot_event"):
                        backend.trigger_hotspot_event(True)

                # Phase 3: 35s - 55s -> Inspect Billboards & Flag Suspect
                elif 35.0 <= sim_time < 55.0:
                    if hasattr(backend, "trigger_hotspot_event"):
                        backend.trigger_hotspot_event(False)
                    if not inspection_triggered:
                        decision_engine.trigger_inspection(duration=10.0)
                        inspection_triggered = True

                # Phase 4: 55s+ -> Battery Depletion & Return Home
                elif sim_time >= 55.0:
                    if hasattr(backend, "battery"):
                        backend.battery = max(5.0, 18.0 - (sim_time - 55.0) * 0.5)

            # -------------------------------------------------------------
            # Simulation & Sensor Step
            # -------------------------------------------------------------
            if not backend.step(dt):
                print("[Mission] Backend stream concluded.")
                break

            frame = backend.get_frame()
            pose = backend.get_pose()
            battery = backend.get_battery()
            gt_pedestrians = backend.get_ground_truth_pedestrians()

            # -------------------------------------------------------------
            # Perception Pipeline (with frame skip for real-time CPU speed)
            # -------------------------------------------------------------
            if frame_idx % frame_skip == 0:
                last_det_result = detector.detect_and_track(frame)
                # Face recognition executes when descending for inspection
                if decision_engine.current_mission == MissionState.INSPECT_FACES and pose.z <= 8.0:
                    last_face_results = face_recognizer.recognize_faces(frame)
                else:
                    last_face_results = []

            # -------------------------------------------------------------
            # Decision Engine
            # -------------------------------------------------------------
            crowd_count = last_det_result.count if last_det_result else 0
            is_hotspot = getattr(last_det_result, "is_hotspot", False)
            hotspot_norm = getattr(last_det_result, "hotspot_centroid", None)

            decision_out = decision_engine.evaluate(
                crowd_count=crowd_count,
                is_hotspot=is_hotspot,
                hotspot_norm=hotspot_norm,
                face_results=last_face_results,
                battery=battery,
                drone_pose=pose,
                dt=dt
            )

            # -------------------------------------------------------------
            # Virtual Arduino Alert Transmission
            # -------------------------------------------------------------
            arduino_pkt = arduino.update_alert(decision_out.alert_level.value)

            # Log alert transition if state changed
            if decision_out.state_changed:
                logger.log_alert(
                    sim_time=sim_time,
                    alert_level=decision_out.alert_level.value,
                    trigger_reason=decision_out.trigger_reason,
                    details=decision_out.details
                )

            # -------------------------------------------------------------
            # Autonomous Mission & Flight Control
            # -------------------------------------------------------------
            mission_manager.update(decision_out, pose, dt=dt)
            motor_cmd = flight_controller.compute_motor_commands(pose, dt=dt)
            backend.send_command(motor_cmd)

            # Synchronize kinematic setpoints for VideoFileBackend
            if hasattr(backend, "set_flight_targets"):
                backend.set_flight_targets(
                    flight_controller.target_x,
                    flight_controller.target_y,
                    flight_controller.target_z,
                    flight_controller.target_yaw
                )

            # -------------------------------------------------------------
            # Telemetry Logging
            # -------------------------------------------------------------
            fps = 1.0 / max(1e-4, time.time() - loop_start)
            logger.log_telemetry(
                sim_time=sim_time,
                frame_idx=frame_idx,
                drone_pose=pose,
                battery=battery,
                crowd_count=crowd_count,
                ground_truth_count=len(gt_pedestrians),
                alert_level=decision_out.alert_level.value,
                mission_state=decision_out.mission_state.value,
                fps=fps
            )

            # -------------------------------------------------------------
            # HUD Dashboard Rendering & Interaction
            # -------------------------------------------------------------
            hud_canvas = dashboard.render(
                frame=frame,
                detection_result=last_det_result,
                face_results=last_face_results,
                decision_out=decision_out,
                arduino=arduino,
                drone_pose=pose,
                fps=fps,
                battery=battery
            )

            if video_writer is not None:
                video_writer.write(hud_canvas)

            if not args.headless:
                key = dashboard.show(hud_canvas, wait_ms=1)
                if key in [27, ord('q'), ord('Q')]: # ESC or 'q'
                    print("[Mission] User quit requested.")
                    break
                elif key in [ord('i'), ord('I')]: # Manual inspection trigger
                    decision_engine.trigger_inspection(duration=6.0)
                    dashboard.add_log("Manual Inspection Command Triggered", (0, 200, 255))

            # Maintain target rate
            elapsed_loop = time.time() - loop_start
            sleep_sec = dt - elapsed_loop
            if sleep_sec > 0:
                time.sleep(sleep_sec)

            frame_idx += 1

    except KeyboardInterrupt:
        print("[Mission] Interrupted by operator.")

    finally:
        backend.disconnect()
        if video_writer is not None:
            video_writer.release()
        if not args.headless:
            dashboard.close()
        logger.save_summary({"mode": args.mode, "fps": target_fps})
        print(f"[Mission] Session complete. Telemetry saved to {logger.telemetry_path}.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI-Based Smart Crowd Monitoring Payload for Surveillance Drones.")
    parser.add_argument("--mode", type=str, default="scenario", choices=["scenario", "video", "webcam", "webots"],
                        help="Execution mode: scenario (auto demo), video, webcam, or webots")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--video", type=str, default="", help="Path to custom video file for --mode video")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera index for --mode webcam")
    parser.add_argument("--duration", type=float, default=65.0, help="Mission duration limit in seconds (0 for infinite)")
    parser.add_argument("--record", type=str, default="", help="Output path to record full HUD video (e.g. results/demo.mp4)")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without GUI window")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_arguments()
    run_mission(cli_args)
