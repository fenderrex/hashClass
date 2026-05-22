@echo off
REM Change directory to the user's Desktop
cd /d %USERPROFILE%\Desktop

REM Create (or update) the virtual environment
python -m venv venv

REM Activate the venv
call venv\Scripts\activate.bat




REM Install required packages
pip install --upgrade pip
pip install pillow opencv-python opencv-contrib-python pyppeteer
pip install matplotlib torch torchvision
REM Run your Python script
chcp 65001
python segnet.py

REM Pause so you can see any output or errors
pause
