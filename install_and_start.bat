@echo off
REM Change directory to the user's Desktop
cd /d %USERPROFILE%\Desktop

REM Create (or update) the virtual environment
python -m venv venv

REM Activate the venv
call venv\Scripts\activate.bat

REM Install required packages
pip install --upgrade pip
pip install pillow opencv-python opencv-contrib-python
pip install pygame PyOpenGL numpy
pip install mapbox-earcut

REM Run your Python script
python star.py abbb1.png

REM Pause so you can see any output or errors
pause
