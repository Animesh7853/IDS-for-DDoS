@echo off
echo Installing required dependencies...
pip install -r requirements.txt
echo.
echo Starting Network Attack Monitoring Dashboard...
echo.
python app.py
echo.
pause
