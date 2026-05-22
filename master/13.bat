@echo off
REM Change directory to the user's Desktop
cd /d %USERPROFILE%\Desktop

REM Create (or update) the virtual environment
python -m venv venv

REM Activate the venv
call venv\Scripts\activate.bat

REM Install required packages

pip install torch torchvision segmentation-models-pytorch albumentations
REM Run your Python script
python 13.py

REM Pause so you can see any output or errors
pause
