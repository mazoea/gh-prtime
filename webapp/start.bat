@echo off
REM Start the GitHub PR Time ETA Dashboard
REM Web interface will be available at http://localhost:5000

echo Starting GitHub PR Time ETA Dashboard...
echo.
echo Web interface will be available at:
echo   http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.

python web_app.py

pause
