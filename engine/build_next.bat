@echo off
rem Builds to build\next\ so the running server's engine is untouched; swap in with the server stopped.
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CUDA=%~dp0..\third_party\cuda
set PATH=%CUDA%\bin;%PATH%
cd /d %~dp0
rem optional modules: the face detector and the upscaler are compiled in when their sources exist
set EXTRA=
if exist src\detect.cu set EXTRA=%EXTRA% -DKILN_FACE src\detect.cu
if exist src\upscale.cu set EXTRA=%EXTRA% -DKILN_UPSCALE src\upscale.cu
if not exist build\next mkdir build\next
nvcc -O3 -std=c++17 -arch=sm_75 -Xcompiler "/EHsc /O2 /utf-8" -o build\next\kiln-engine.exe ^
  src\main.cpp src\safetensors.cpp src\kernels.cu src\te.cu src\dit.cu src\vae.cu src\pipeline.cu src\graph.cpp src\sdops.cu src\sdxl.cu src\clip.cu src\sdvae.cu src\sdpipe.cu %EXTRA% -lcublas || exit /b 1
echo BUILD OK
