@echo off
echo =================================================
echo Diabetes Classification Case Study - Runner Script
echo =================================================
echo.

REM Set the correct working directory
cd /d "%~dp0"
echo Working directory: %CD%

:menu
echo Choose an option:
echo [1] Run the diabetes classification study
echo [2] Make predictions using trained model
echo [3] Exit
echo.

set /p option="Enter option (1-3): "

if "%option%"=="1" (
    echo.
    echo Running the diabetes classification study...
    python diabetic_classification_simple.py
    echo.
    echo Press any key to return to menu...
    pause > nul
    cls
    goto menu
) else if "%option%"=="2" (
    echo.
    echo Running the prediction tool...
    python predict_diabetes.py
    echo.
    echo Press any key to return to menu...
    pause > nul
    cls
    goto menu
) else if "%option%"=="3" (
    echo Exiting...
    exit
) else (
    echo Invalid option. Please try again.
    echo.
    goto menu
)
