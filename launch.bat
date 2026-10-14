@echo off
title Drone Surveillance Payload - Mission Launcher
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" launch.py %*
) else (
    python launch.py %*
)
pause

