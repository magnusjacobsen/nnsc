@echo off
call conda activate baenv
call python -m pytest
pause