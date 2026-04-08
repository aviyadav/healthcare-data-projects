@echo off
REM ============================================================
REM  Clinical Data API – Windows convenience launcher
REM
REM  Usage:
REM    start_server.bat
REM
REM  Override defaults before calling the script:
REM    set CLINICAL_WORKERS=4
REM    set CLINICAL_BIND=0.0.0.0:8090
REM    set CLINICAL_LOG_LEVEL=info
REM    set CLINICAL_TIMEOUT=120
REM    start_server.bat
REM
REM  Or inline (CMD):
REM    set CLINICAL_WORKERS=4 && start_server.bat
REM
REM  PowerShell equivalent:
REM    $env:CLINICAL_WORKERS=4; .\start_server.bat
REM ============================================================

setlocal

REM ── Change to the directory containing this script ─────────────────────────
cd /d "%~dp0"

REM ── Check Python is available ──────────────────────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo [start_server] ERROR: 'python' not found on PATH.
    echo        Install Python 3.11+ and make sure it is added to PATH.
    exit /b 1
)

REM ── Activate virtual environment if present ────────────────────────────────
if exist ".venv\Scripts\activate.bat" (
    echo [start_server] Activating virtual environment: .venv
    call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    echo [start_server] Activating virtual environment: venv
    call "venv\Scripts\activate.bat"
) else (
    echo [start_server] No virtual environment found – using system Python.
    echo        Tip: create one with:  python -m venv .venv
)

REM ── Verify the launcher script exists ─────────────────────────────────────
if not exist "start_server.py" (
    echo [start_server] ERROR: start_server.py not found in "%CD%".
    echo        Run this .bat file from the clinical_data_api\ directory.
    exit /b 1
)

REM ── Apply defaults for any variables not already set by the caller ─────────
REM    (setlocal above ensures these do not leak into the parent shell)
if not defined CLINICAL_BIND      set CLINICAL_BIND=0.0.0.0:8090
if not defined CLINICAL_LOG_LEVEL set CLINICAL_LOG_LEVEL=info
if not defined CLINICAL_TIMEOUT   set CLINICAL_TIMEOUT=120

REM CLINICAL_WORKERS: leave unset here so start_server.py can apply the
REM (2 * CPU_COUNT) + 1 heuristic if the caller has not specified a value.

REM ── Print effective configuration ─────────────────────────────────────────
echo.
echo [start_server] ===================================================
echo [start_server]  Clinical Data API
echo [start_server] ---------------------------------------------------
echo [start_server]  Bind        : %CLINICAL_BIND%
echo [start_server]  Log level   : %CLINICAL_LOG_LEVEL%
echo [start_server]  Timeout     : %CLINICAL_TIMEOUT%s
if defined CLINICAL_WORKERS (
    echo [start_server]  Workers     : %CLINICAL_WORKERS%
) else (
    echo [start_server]  Workers     : auto ^(2 x CPU + 1^)
)
echo [start_server] ===================================================
echo.

REM ── Launch ─────────────────────────────────────────────────────────────────
python start_server.py
set EXIT_CODE=%ERRORLEVEL%

REM ── Report exit status ────────────────────────────────────────────────────
if %EXIT_CODE% neq 0 (
    echo.
    echo [start_server] Server exited with code %EXIT_CODE%.
)

endlocal
exit /b %EXIT_CODE%
