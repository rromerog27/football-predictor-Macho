@echo off
setlocal
rem Programa una instantánea del mercado de FC 27 cada 30 minutos con el
rem Programador de tareas de Windows. Doble clic para activarlo; para quitarlo,
rem doble clic en quitar_snapshots_windows.bat.
rem Usa pythonw.exe del entorno .venv: se ejecuta sin abrir ninguna ventana.

set "PROJECT=%~dp0.."
for %%I in ("%PROJECT%") do set "PROJECT=%%~fI"
set "PYW=%PROJECT%\.venv\Scripts\pythonw.exe"
set "SCRIPT=%PROJECT%\scripts\fc27_snapshot.py"

if not exist "%PYW%" (
    echo No se encuentra "%PYW%".
    echo Primero crea el entorno e instala la app:
    echo     python -m venv .venv
    echo     .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

schtasks /Create /F /SC MINUTE /MO 30 /TN "FC27 Mercado - instantanea" /TR "\"%PYW%\" \"%SCRIPT%\""
if errorlevel 1 (
    echo No se pudo crear la tarea programada.
    pause
    exit /b 1
)

echo Haciendo la primera instantanea ahora...
"%PROJECT%\.venv\Scripts\python.exe" "%SCRIPT%"

echo.
echo Listo: se guardara una instantanea cada 30 minutos mientras tu sesion de Windows este abierta.
echo Puedes revisar que funciona en: %PROJECT%\data\fc27_snapshot.log
pause
