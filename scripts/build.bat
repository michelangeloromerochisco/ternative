@echo off
setlocal enabledelayedexpansion

rem If not running inside a Visual Studio Developer Command Prompt,
rem delegate to the PowerShell auto-detect script.
if "%VSCMD_VER%"=="" (
    echo Not in a Visual Studio Developer Command Prompt.
    echo Delegating to auto-detect PowerShell script...
    powershell -ExecutionPolicy Bypass -File "%~dp0build.ps1"
    exit /b %ERRORLEVEL%
)

set "BUILD_DIR=build"
set "BUILD_TYPE=Release"
set "CUDA=OFF"

:parse_args
if "%~1"=="" goto :done_parsing
if /i "%~1"=="--cuda" (
    set "CUDA=ON"
    shift
    goto :parse_args
)
if /i "%~1"=="--debug" (
    set "BUILD_TYPE=Debug"
    shift
    goto :parse_args
)
shift
goto :parse_args
:done_parsing

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
cd "%BUILD_DIR%"

echo Configuring ternative.cpp (CUDA=%CUDA%, BUILD_TYPE=%BUILD_TYPE%)...
cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DTERNATIVE_CUDA=%CUDA% ^
    -DTERNATIVE_BUILD_TESTS=ON ^
    -DCMAKE_CUDA_ARCHITECTURES=86

if errorlevel 1 (
    echo CMake configuration failed.
    exit /b 1
)

echo Building...
cmake --build . --config %BUILD_TYPE% --parallel %NUMBER_OF_PROCESSORS%

if errorlevel 1 (
    echo Build failed.
    exit /b 1
)

echo.
echo Build complete. Binary: %BUILD_DIR%\%BUILD_TYPE%\ternative.exe

if exist "%BUILD_DIR%\%BUILD_TYPE%\ternative_test.exe" (
    echo.
    echo Running tests...
    "%BUILD_DIR%\%BUILD_TYPE%\ternative_test.exe"
)
