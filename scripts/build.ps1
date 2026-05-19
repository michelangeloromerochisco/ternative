#!/usr/bin/env pwsh
# Auto-detect build script for ternative.cpp on Windows
# Finds MSVC 2022, clang-cl, or falls back to MinGW

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Split-Path -Parent $scriptDir
$buildDir = Join-Path $projectRoot "build"

function Find-MSVC {
    # Try to locate VS 2022 through vswhere
    $vswherePaths = @(
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe",
        "$env:ProgramFiles\Microsoft Visual Studio\Installer\vswhere.exe"
    )

    foreach ($vp in $vswherePaths) {
        if (Test-Path $vp) {
            $installPath = & $vp -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
            if ($installPath) {
                $vcvars = Join-Path $installPath "VC\Auxiliary\Build\vcvars64.bat"
                if (Test-Path $vcvars) {
                    return $installPath
                }
            }
        }
    }

    # Try registry
    $regPaths = @(
        "HKLM:\SOFTWARE\Microsoft\VisualStudio\17.0",
        "HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\17.0"
    )
    foreach ($rp in $regPaths) {
        if (Test-Path $rp) {
            $installPath = (Get-ItemProperty $rp -Name "InstallDir" -ErrorAction SilentlyContinue).InstallDir
            if ($installPath) {
                $vcvars = Join-Path $installPath "..\..\VC\Auxiliary\Build\vcvars64.bat"
                $vcvars = (Resolve-Path $vcvars -ErrorAction SilentlyContinue).Path
                if ($vcvars -and (Test-Path $vcvars)) {
                    return (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $vcvars)))
                }
            }
        }
    }

    # Try common paths (including Build Tools)
    $commonPaths = @(
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Professional",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Enterprise",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\Community",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\Professional",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\Enterprise",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\BuildTools"
    )
    foreach ($cp in $commonPaths) {
        $vcvars = Join-Path $cp "VC\Auxiliary\Build\vcvars64.bat"
        if (Test-Path $vcvars) {
            return $cp
        }
    }

    return $null
}

function Invoke-MSVCBuild {
    param($VsPath)

    Write-Host "Found Visual Studio 2022 at: $VsPath" -ForegroundColor Green

    # Run vcvars64 in a child process and capture environment
    $vcvars = Join-Path $VsPath "VC\Auxiliary\Build\vcvars64.bat"
    $tempBat = [System.IO.Path]::GetTempFileName() + ".bat"

    @"
@echo off
call "$vcvars" >nul 2>&1
cd /d "$projectRoot"
if not exist build mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DTERNATIVE_BUILD_TESTS=ON -DCMAKE_EXE_LINKER_FLAGS="/MANIFEST:NO" -DCMAKE_SHARED_LINKER_FLAGS="/MANIFEST:NO"
cmake --build . --config Release --parallel
"@ | Out-File -FilePath $tempBat -Encoding ASCII

    Write-Host "Building with MSVC..." -ForegroundColor Cyan
    cmd /c $tempBat
    Remove-Item $tempBat -ErrorAction SilentlyContinue

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed." -ForegroundColor Red
        exit 1
    }

    Write-Host "Build successful!" -ForegroundColor Green
    $exe = Join-Path $buildDir "Release\ternative.exe"
    if (Test-Path $exe) {
        Write-Host "Executable: $exe" -ForegroundColor Green
    }

    $testExe = Join-Path $buildDir "Release\ternative_test.exe"
    if (Test-Path $testExe) {
        Write-Host "Running tests..." -ForegroundColor Cyan
        & $testExe
    }
}

function Invoke-ClangBuild {
    Write-Host "Attempting clang-cl build..." -ForegroundColor Cyan

    $clang = Get-Command clang-cl -ErrorAction SilentlyContinue
    if (-not $clang) {
        Write-Host "clang-cl not found in PATH." -ForegroundColor Yellow
        return $false
    }

    Set-Location $projectRoot
    if (-not (Test-Path $buildDir)) { New-Item -ItemType Directory -Path $buildDir | Out-Null }
    Set-Location $buildDir

    cmake .. -G "Ninja" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DTERNATIVE_BUILD_TESTS=ON
    cmake --build . --parallel

    if ($LASTEXITCODE -ne 0) {
        Write-Host "clang-cl build failed." -ForegroundColor Red
        return $false
    }

    Write-Host "clang-cl build successful!" -ForegroundColor Green
    return $true
}

function Invoke-MinGWBuild {
    Write-Host "Attempting MinGW build..." -ForegroundColor Cyan

    $gcc = Get-Command g++ -ErrorAction SilentlyContinue
    if (-not $gcc) {
        Write-Host "g++ not found in PATH." -ForegroundColor Yellow
        return $false
    }

    Set-Location $projectRoot
    if (-not (Test-Path $buildDir)) { New-Item -ItemType Directory -Path $buildDir | Out-Null }
    Set-Location $buildDir

    cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DTERNATIVE_BUILD_TESTS=ON
    cmake --build . --parallel

    if ($LASTEXITCODE -ne 0) {
        Write-Host "MinGW build failed." -ForegroundColor Red
        return $false
    }

    Write-Host "MinGW build successful!" -ForegroundColor Green
    return $true
}

# Main logic
Write-Host "=== ternative.cpp Windows Build ===" -ForegroundColor Cyan
Write-Host "Project root: $projectRoot" -ForegroundColor Gray

$vsPath = Find-MSVC
if ($vsPath) {
    Invoke-MSVCBuild -VsPath $vsPath
    exit 0
}

Write-Host "Visual Studio 2022 not found. Trying alternatives..." -ForegroundColor Yellow

if (Invoke-ClangBuild) { exit 0 }
if (Invoke-MinGWBuild) { exit 0 }

Write-Host @"
ERROR: No suitable compiler found.

Please install one of:
  1. Visual Studio 2022 Community (free) with "Desktop development with C++" workload
     https://visualstudio.microsoft.com/downloads/

  2. LLVM/Clang for Windows
     https://github.com/llvm/llvm-project/releases

  3. MinGW-w64 via MSYS2
     https://www.msys2.org/
"@ -ForegroundColor Red

exit 1
