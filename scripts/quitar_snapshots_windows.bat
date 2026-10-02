@echo off
rem Quita la tarea programada que crea programar_snapshots_windows.bat.
schtasks /Delete /F /TN "FC27 Mercado - instantanea"
if errorlevel 1 (
    echo No habia ninguna tarea programada de FC27 Mercado.
) else (
    echo Listo: ya no se guardaran instantaneas automaticas.
)
pause
