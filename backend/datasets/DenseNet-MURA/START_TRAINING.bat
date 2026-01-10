@echo off
REM ========================================
REM MURA FINAL TRAINING - QUICK START
REM ========================================

cd /d %~dp0

echo.
echo ================================================================================
echo  MURA Training - DenseNet-169
echo  Musculoskeletal Radiograph Abnormality Detection
echo ================================================================================
echo.

REM Check dataset first
echo [1/3] Checking dataset...
python train_mura_final.py --check_only
if errorlevel 1 (
    echo.
    echo ERROR: Dataset check failed!
    echo Please ensure MURA dataset is in one of these locations:
    echo   - MURA/
    echo   - MURA-v1.1/
    echo   - MURA-v1.0/
    echo.
    pause
    exit /b 1
)

echo.
echo [2/3] Dataset verified successfully!
echo.
echo [3/3] Starting training...
echo.
echo Training Configuration:
echo   Study Type: XR_WRIST (fastest for testing)
echo   Epochs: 10
echo   Estimated Time: ~30-60 minutes (CPU) or ~5-10 minutes (GPU)
echo.
echo Press Ctrl+C to cancel, or
pause

REM Start training
python train_mura_final.py --study_type XR_WRIST --epochs 10

echo.
echo ================================================================================
echo  TRAINING COMPLETE
echo ================================================================================
echo.
echo Model saved to: models/XR_WRIST/model.pth
echo.
echo Next steps:
echo   1. Train more epochs for better performance:
echo      python train_mura_final.py --study_type XR_WRIST --epochs 20
echo.
echo   2. Train other body parts:
echo      python train_mura_final.py --study_type XR_SHOULDER --epochs 15
echo.
echo   3. Train all study types:
echo      python train_mura_final.py --study_type all --epochs 15
echo.
pause
