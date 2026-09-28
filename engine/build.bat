@echo off
setlocal
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (echo Visual Studio 2022 C++ build tools were not found. & exit /b 1)
rem vcvars64.bat calls vswhere.exe itself; keep it findable so it does not print an error
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "%VCVARS%" >nul
set CUDA=%~dp0..\third_party\cuda
set PATH=%CUDA%\bin;%PATH%
cd /d %~dp0
rem optional modules: the face detector and the upscaler are compiled in when their sources exist
set EXTRA=
if exist src\detect.cu set EXTRA=%EXTRA% -DKILN_FACE src\detect.cu
if exist src\upscale.cu set EXTRA=%EXTRA% -DKILN_UPSCALE src\upscale.cu
if not exist build mkdir build
rem GPU code: Turing (sm_75), Ampere (sm_80, also runs on RTX 30), Ada (sm_89, RTX 40), Blackwell (sm_120, RTX 50),
rem plus PTX that newer GPUs compile on first start. The engine picks tensor-core or CUDA-core kernels at startup.
set ARCH=-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_80,code=compute_80
nvcc -O3 -std=c++17 %ARCH% --threads 0 -Xcompiler "/EHsc /O2 /utf-8" -o build\kiln-engine.exe ^
  src\main.cpp src\safetensors.cpp src\kernels.cu src\te.cu src\dit.cu src\vae.cu src\pipeline.cu src\graph.cpp src\sdops.cu src\sdxl.cu src\clip.cu src\sdvae.cu src\sdpipe.cu %EXTRA% -lcublas || exit /b 1
copy /y "%CUDA%\bin\x64\cudart64_13.dll" build\ >nul
copy /y "%CUDA%\bin\x64\cublas64_13.dll" build\ >nul
copy /y "%CUDA%\bin\x64\cublasLt64_13.dll" build\ >nul
echo BUILD OK
