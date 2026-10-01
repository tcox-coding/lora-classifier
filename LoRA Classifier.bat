@echo off
rem Opens the LoRA Classifier UI (no console window).
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" -m lora_classifier
