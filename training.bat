@echo off
setlocal
title CYN-X Vision - Dataset Pipeline

echo.
echo ==========================================
echo        CYN-X VISION DATASET PIPELINE
echo ==========================================
echo.

cd /d "%~dp0"

REM ------------------------------------------
REM Check Python
REM ------------------------------------------
echo [1/6] Checking Python...
python --version
if errorlevel 1 (
    echo.
    echo ERROR: Python was not found.
    pause
    exit /b 1
)

echo.

REM ------------------------------------------
REM Scrape new images
REM ------------------------------------------
echo [2/6] Scraping images...
python .\tools\dataset_scraper.py --count 100 --out dataset

if errorlevel 1 (
    echo.
    echo ERROR: Image scraper failed.
    pause
    exit /b 1
)

echo.

REM ------------------------------------------
REM Organize / normalize images
REM ------------------------------------------
echo [3/6] Organizing images...
python .\tools\organize_images.py

if errorlevel 1 (
    echo.
    echo ERROR: Image organizer failed.
    pause
    exit /b 1
)

echo.

REM ------------------------------------------
REM Auto-label using current trained model
REM ------------------------------------------
echo [4/6] Auto-labeling new images...

python tools\auto_label.py ^
    --model "runs\detect\runs\weed_pen_bootstrap\weights\best.pt" ^
    --images "dataset\organized" ^
    --labels "dataset\labels" ^
    --review "dataset\review"

if errorlevel 1 (
    echo.
    echo ERROR: Auto-labeling failed.
    pause
    exit /b 1
)

echo.

REM ------------------------------------------
REM Prepare YOLO training dataset
REM ------------------------------------------
echo [5/6] Preparing YOLO dataset...

python tools\prepare_dataset.py

if errorlevel 1 (
    echo.
    echo ERROR: Dataset preparation failed.
    pause
    exit /b 1
)

echo.

REM ------------------------------------------
REM Show dataset statistics
REM ------------------------------------------
echo [6/6] Dataset statistics...
echo.

echo Images:
dir /b dataset\organized\*.jpg 2^>nul | find /c /v ""

echo.
echo Labels:
dir /b dataset\labels\*.txt 2^>nul | find /c /v ""

echo.
echo ==========================================
echo          PIPELINE COMPLETE
echo ==========================================
echo.
echo Review auto-labels in:
echo   dataset\review
echo.
echo Training dataset:
echo   dataset\yolo
echo.
echo Current model:
echo   runs\detect\runs\weed_pen_bootstrap\weights\best.pt
echo.

pause