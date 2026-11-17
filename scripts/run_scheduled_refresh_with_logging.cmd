@echo off
setlocal
cd /d "D:\Vedant\Codes\Landslide-India"

set "LOG=outputs\refresh_reports\scheduled_task_console.log"
set "PYTHONUNBUFFERED=1"

echo.>> "%LOG%"
echo ==========================================>> "%LOG%"
echo START: %DATE% %TIME%>> "%LOG%"

".venv\Scripts\python.exe" -u -X utf8 -B -m backend.run_scheduled_refresh --max-latency-hours 12.0 --workers 4 >> "%LOG%" 2>&1

set "RC=%ERRORLEVEL%"
echo EXIT CODE: %RC%>> "%LOG%"
echo END: %DATE% %TIME%>> "%LOG%"
exit /b %RC%
