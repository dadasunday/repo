@echo off
REM Full Audio Quality Improvement Pipeline
REM This will process all your data for maximum improvement

echo ================================================================================
echo FULL AUDIO QUALITY IMPROVEMENT PIPELINE
echo ================================================================================
echo.
echo This will:
echo   1. Remaster ALL 224 training segments
echo   2. Post-process ALL production outputs with proper references
echo   3. Run quality control validation
echo.
echo Estimated time: 20-30 minutes
echo.
pause

echo.
echo ================================================================================
echo STEP 1: Remastering Training Data (224 segments)
echo ================================================================================
echo.
python remaster_training_data.py
if %errorlevel% neq 0 (
    echo ERROR: Training data remastering failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo STEP 2: Post-Processing Production Outputs (with remastered references)
echo ================================================================================
echo.
python production_post_process.py
if %errorlevel% neq 0 (
    echo ERROR: Production post-processing failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo STEP 3: Quality Control Validation
echo ================================================================================
echo.
python audio_quality_checker.py --compare training_data/segments_remastered production_clone_output_processed

echo.
echo ================================================================================
echo PIPELINE COMPLETE!
echo ================================================================================
echo.
echo Next steps:
echo   1. Listen to production_clone_output_processed/ files
echo   2. Compare with originals in production_clone_output/
echo   3. If satisfied, update voice cloning scripts to use segments_remastered/
echo   4. Regenerate all voice samples
echo.
pause
