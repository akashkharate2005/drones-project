@echo off
echo ==============================================================
echo Connecting AI Payload to Webots (Mavic 2 Pro EXTERN Controller)...
echo ==============================================================
echo 1. Ensure Webots is open with worlds/crowd_plaza.wbt loaded.
echo 2. Verify MAVIC_2_PRO robot controller is set to ^<extern^>.
echo 3. Press Play in Webots simulation toolbar.
echo ==============================================================
set WEBOTS_CONTROLLER_URL=ipc://1270/mavic_2_pro
python main.py --mode webots
pause
