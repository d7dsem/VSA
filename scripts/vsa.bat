@echo off
setlocal enabledelayedexpansion

REM Usage guard
if "%~1"=="" (
  echo "Usage: vsa.bat filename.wav|bin [--samp-rate]"
  exit /b 1
)

if not exist "%~f1" (
  echo [ERROR] File not found: [%~f1]
  exit /b 1
)

REM Resolve wrapper dir
set "SCRIPT_DIR=%~dp0"
set "INPUT=%~f1"

REM ============================================================
REM Detect Python: portable or system
REM ============================================================
set "PYTHON_EXE=python"

if exist "%SCRIPT_DIR%python_distr\python.exe" (
  echo [INFO] Using portable Python
  set "PYTHON_EXE=%SCRIPT_DIR%python_distr\python.exe"
  REM Add project dir to PYTHONPATH for portable Python
  set "PYTHONPATH=%SCRIPT_DIR%"
) else (
  echo [INFO] Using system Python
)

REM Collect all arguments except first one
set "EXTRA_ARGS="
shift
:loop
if "%~1"=="" goto endloop
set "EXTRA_ARGS=!EXTRA_ARGS! %~1"
shift
goto loop
:endloop

REM Change to script directory
pushd "%SCRIPT_DIR%\..\src" || (
  echo [ERROR] Failed to cd to "%SCRIPT_DIR%"
  exit /b 2
)

REM Run with input and extra args


::"%PYTHON_EXE%" vsa_file_entry_mk2.py -i "%INPUT%" !EXTRA_ARGS!
"%PYTHON_EXE%" analize_stage_II.py -i "%INPUT%" !EXTRA_ARGS!

set "rc=%ERRORLEVEL%"
popd
exit /b %rc%