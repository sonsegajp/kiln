@echo off
rem usage: run_job.bat <job.json> [extra engine args]   -- one-shot engine run for testing
set PATH=%~dp0build;%PATH%
"%~dp0build\kiln-engine.exe" --models "%~dp0..\models" %2 %3 %4 %5 < "%~1"
