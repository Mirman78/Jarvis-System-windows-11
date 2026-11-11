@echo off
setlocal EnableExtensions

title J.A.R.V.I.S. Launcher

REM Jump to the folder this .bat lives in
cd /d "%~dp0"

REM --- Sanity checks -----------------------------------------------------
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] venv not found in %cd%
    echo.
    echo The venv was created by Jarvis at: C:\Users\ME\Desktop\Another Ai Claude\venv
    echo Ensure the folder exists.
    pause
    exit /b 1
)

if not exist "package.json" (
    echo [ERROR] package.json not found in %cd%
    pause
    exit /b 1
)

REM --- Activate the venv -------------------------------------------------
call "venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate venv
    pause
    exit /b 1
)

REM --- Belt-and-suspenders: force venv\Scripts to the front of PATH -----
set "PATH=%CD%\venv\Scripts;%PATH%"

echo.
echo ============================================================
echo  J.A.R.V.I.S. Launcher (DirectML Enabled)
echo ============================================================
echo  Directory : %CD%
echo  Python    :
where python
echo  Pip       :
where pip
echo ============================================================
echo.

REM --- First-run dependency install --------------------------------------
if not exist "node_modules" (
    echo First run detected - installing npm dependencies...
    call npm install
    if errorlevel 1 (
        echo.
        echo [ERROR] npm install failed
        pause
        exit /b 1
    )
)

REM --- Launch Electron ---------------------------------------------------
echo Starting Electron...
echo.
call npm start

set EXITCODE=%errorlevel%

echo.
echo ============================================================
echo  npm start exited with code %EXITCODE%
echo ============================================================

pause
endlocal
exit /b %EXITCODE%