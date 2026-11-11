@echo off
REM ============================================================
REM  JARVIS HUD — Python runtime setup (system Python 3.12)
REM  Run once after fresh Python install. Idempotent.
REM ============================================================
setlocal

cd /d "%~dp0"

python -V >nul 2>&1
if errorlevel 1 (
  echo [ERROR] python not on PATH. Install Python 3.12 from python.org.
  echo         During install, tick "Add python.exe to PATH".
  exit /b 1
)

python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3,12) else 1)"
if errorlevel 1 (
  echo [ERROR] Python 3.12 required, found:
  python -V
  exit /b 1
)

echo [1/4] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo [2/4] Installing Jarvis requirements...
python -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo [3/4] Installing Playwright browser...
python -m playwright install chromium
if errorlevel 1 goto :fail

echo [4/4] Smoke test...
python test_install.py
if errorlevel 1 (
  echo [WARN] test_install.py failed - check output above.
  exit /b 2
)

echo.
echo [OK] Python environment ready.
exit /b 0

:fail
echo [ERROR] Installation failed. See output above.
exit /b 1