@echo off
setlocal
chcp 65001 >nul

echo ====================================================================
echo     Mini Multimodal Transformer Launcher
echo ====================================================================
echo.

:: Detect Python launcher or executable
set PYTHON_CMD=py -3.13
%PYTHON_CMD% --version >nul 2>&1
if errorlevel 1 (
    set PYTHON_CMD=python
    %PYTHON_CMD% --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python not found in PATH or launcher.
        echo Please ensure Python 3.8+ is installed.
        pause
        exit /b 1
    )
)

if "%1"=="test" (
    %PYTHON_CMD% main.py --mode test
    goto end
)

if "%1"=="web" (
    %PYTHON_CMD% main.py --mode web
    goto end
)

if "%1"=="caption" (
    %PYTHON_CMD% main.py --mode caption
    goto end
)

if "%1"=="vqa" (
    %PYTHON_CMD% main.py --mode vqa
    goto end
)

%PYTHON_CMD% main.py

:end
endlocal
