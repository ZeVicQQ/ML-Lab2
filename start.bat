@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================
echo   Запуск лабораторной работы №2 (Регрессия)
echo ============================================

REM --- Проверка Python ---
where python >nul 2>nul
if errorlevel 1 (
    echo.
    echo [ОШИБКА] Python не найден.
    pause
    exit /b 1
)

REM --- Создание виртуального окружения, если нет ---
if not exist ".venv\Scripts\python.exe" (
    echo Создание виртуального окружения .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ОШИБКА] Не удалось создать venv.
        pause
        exit /b 1
    )
)

REM --- Установка зависимостей, если не установлены ---
".venv\Scripts\python.exe" -c "import numpy, sklearn, matplotlib" >nul 2>nul
if errorlevel 1 (
    echo Установка зависимостей...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    REM Добавлен флаг --no-cache-dir, чтобы не использовать старые битые файлы
    ".venv\Scripts\python.exe" -m pip install --no-cache-dir -r requirements.txt
    if errorlevel 1 (
        echo [ОШИБКА] Не удалось установить зависимости.
        pause
        exit /b 1
    )
)

REM --- Запуск ---
echo.
echo Запуск lab2.py ...
".venv\Scripts\python.exe" lab2.py

echo.
echo Готово. Проверьте папку lab2_results.
pause