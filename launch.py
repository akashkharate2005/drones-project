"""Unified Interactive Launcher for AI-Based Smart Crowd Monitoring Payload.

Provides a 1-click terminal menu that automatically detects the project
virtual environment (.venv) and launches all operational modes, tests,
evaluations, and presentation dashboards.
"""
import os
import sys
import subprocess
import webbrowser

# Determine appropriate Python executable (prioritize local .venv)
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
VENV_PYTHON_WIN = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
VENV_PYTHON_UNIX = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")

if os.name == "nt" and os.path.isfile(VENV_PYTHON_WIN):
    PYTHON_EXE = VENV_PYTHON_WIN
elif os.path.isfile(VENV_PYTHON_UNIX):
    PYTHON_EXE = VENV_PYTHON_UNIX
else:
    PYTHON_EXE = sys.executable

BANNER = r"""
================================================================================
  ___   ___       ___ _   _ ______     _______ ___ _     _        _    _   _ 
 / _ \ |_ _|     / ___| | | |  _ \ \   / / ____|_ _| |   | |      / \  | \ | |
| | | | | | ____| |   | |_| | |_) \ \ / /|  _|  | || |   | |     / _ \ |  \| |
| |_| | | ||____| |___|  _  |  _ < \ V / | |___ | || |___| |___ / ___ \| |\  |
 \___/ |___|     \____|_| |_|_| \_\ \_/  |_____|___|_____|_____/_/   \_\_| \_|
       P A Y L O A D   M I S S I O N   C O N T R O L   L A U N C H E R
================================================================================
  System: Drone Aerial Surveillance + Crowd Density + Face Recognition
  Platform: Webots / Video / Webcam / SITL MAVLink | Python: {py_info}
================================================================================
"""


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def run_command(cmd, wait=True):
    print(f"\n[Launcher] Executing: {' '.join(cmd)}\n" + "-" * 70)
    try:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
        return proc.returncode
    except KeyboardInterrupt:
        print("\n[Launcher] Process stopped by user.")
        return 0
    except Exception as e:
        print(f"\n[Launcher] Error executing command: {e}")
        return 1


def open_showcase_in_browser():
    showcase_path = os.path.join(PROJECT_ROOT, "showcase.html")
    if os.path.exists(showcase_path):
        print(f"\n[Launcher] Opening Mission Control & Presentation Showcase in browser...")
        webbrowser.open(f"file://{showcase_path}")
    else:
        print(f"[Launcher] Error: showcase.html not found at {showcase_path}")


def print_menu():
    print("""
  OPERATIONAL MODES & PRESENTATION MENU:
  ----------------------------------------------------------------------------
  [1] Run Full 4-Phase Autonomous Scenario (OpenCV Tactical HUD)
      -> Patrol -> Crowd Hotspot Orbit -> Face Inspection Descent -> RTH
  [2] Run Multi-Person Face Recognition Demo (InsightFace Grid)
      -> Test images gallery (Akash, Laksh, Ritu) with bounding boxes
  [3] Run Video Surveillance Processing Mode
      -> Aerial drone surveillance video with YOLOv8 + ByteTrack
  [4] Run Live USB Webcam Bench Test Mode
      -> Real-time tracking & alert engine on camera index 0
  [5] Run Quantitative Evaluation & Generate Benchmark Plots
      -> Computes MAE, RMSE, Precision, Recall, Latency & updates results/
  [6] Run Offline Unit Test Suite (pytest)
      -> Runs 19 modular tests covering perception, decision, flight, alerts
  [7] Open Interactive Web Presentation & Mission Control
      -> Self-contained browser dashboard (showcase.html)
  [8] Regenerate Webots World & Synthetic Plaza Assets
      -> Builds crowd_plaza.wbt & synthetic data
  [9] Run Fast Headless Scenario Benchmark (Simulation check)
      -> 65-second benchmark without opening OpenCV GUI
  [Q] Exit Launcher
  ----------------------------------------------------------------------------
""")


def main():
    if len(sys.argv) > 1:
        choice = sys.argv[1].strip()
        handle_choice(choice)
        return

    while True:
        clear_screen()
        print(BANNER.format(py_info=os.path.basename(os.path.dirname(os.path.dirname(PYTHON_EXE))) or "System"))
        print_menu()
        try:
            choice = input("  Select an option [1-9, Q]: ").strip().upper()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if choice in ("Q", "QUIT", "EXIT"):
            print("\nExiting Launcher. Safe flying!")
            break

        handle_choice(choice)
        input("\nPress ENTER to return to menu...")


def handle_choice(choice):
    if choice == "1":
        run_command([PYTHON_EXE, "main.py", "--mode", "scenario", "--duration", "65"])
    elif choice == "2":
        run_command([PYTHON_EXE, "face_demo.py"])
    elif choice == "3":
        run_command([PYTHON_EXE, "main.py", "--mode", "video", "--video", "data/samples/crowd_surveillance_demo.mp4"])
    elif choice == "4":
        run_command([PYTHON_EXE, "main.py", "--mode", "webcam", "--camera-id", "0"])
    elif choice == "5":
        run_command([PYTHON_EXE, "evaluate.py"])
    elif choice == "6":
        pytest_exe = os.path.join(os.path.dirname(PYTHON_EXE), "pytest.exe")
        if os.path.isfile(pytest_exe):
            run_command([pytest_exe, "tests/", "-v"])
        else:
            run_command([PYTHON_EXE, "-m", "pytest", "tests/", "-v"])
    elif choice == "7":
        open_showcase_in_browser()
    elif choice == "8":
        run_command([PYTHON_EXE, "scripts/generate_synthetic_data.py"])
        run_command([PYTHON_EXE, "scripts/generate_world.py", "--crowd-size", "45", "--density", "medium"])
    elif choice == "9":
        run_command([PYTHON_EXE, "main.py", "--mode", "scenario", "--duration", "65", "--headless"])
    else:
        print(f"\n[Launcher] Invalid selection: {choice}")


if __name__ == "__main__":
    main()

