@echo off
cd %~dp0
pip install -r requirements.txt
pip install openpyxl
python main.py
pause