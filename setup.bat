@echo off
REM Setup script for CandiSight ML Project on Windows

echo Creating Python Virtual Environment...
python -m venv venv

echo.
echo Activating Virtual Environment...
call venv\Scripts\activate.bat

echo.
echo Installing Required Packages...
pip install -r requirements.txt

echo.
echo Setup Complete! Virtual environment is ready.
echo.
echo To activate the virtual environment in the future, run:
echo   venv\Scripts\activate.bat
echo.
echo To deactivate, run:
echo   deactivate
echo.
pause
