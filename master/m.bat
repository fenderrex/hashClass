@echo off
REM Change directory to the user's Desktop
cd /d "%~dp0"

REM Create (or update) the virtual environment
python -m venv venv


if not exist "venv" (
    python -m venv venv
)

call venv\Scripts\activate.bat
REM Install required packages
pip install pyppeteer
pip install pillow opencv-python opencv-contrib-python pyppeteer
pip install matplotlib
REM Run your Python script
python m.py

REM Pause so you can see any output or errors
pause
