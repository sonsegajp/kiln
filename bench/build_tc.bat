@echo off
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CUDA=%~dp0..\third_party\cuda
set PATH=%CUDA%\bin;%CUDA%\bin\x64;%PATH%
cd /d %~dp0
nvcc -O3 -std=c++17 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -Xptxas -v -o %1.exe %1.cu -lcublas || exit /b 1
