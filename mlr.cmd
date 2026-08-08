@echo off
rem Launch the mlr menu from anywhere: double-click this file, or run
rem `mlr.cmd` (optionally with subcommands, e.g. `mlr.cmd runs`).
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" ml-research
cd /d "%~dp0"
mlr %*
if errorlevel 1 pause
