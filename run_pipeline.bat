@echo off
chcp 65001 >nul
echo ==========================================================
echo   NASDAQ MACRO SHOCKS ANALYSIS - AUTOMATED PIPELINE
echo ==========================================================
echo.
echo Kich hoat moi truong ao...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo [INFO] Da kich hoat moi truong: .venv
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [INFO] Da kich hoat moi truong: venv
) else if exist "env\Scripts\activate.bat" (
    call env\Scripts\activate.bat
    echo [INFO] Da kich hoat moi truong: env
) else if exist ".env\Scripts\activate.bat" (
    call .env\Scripts\activate.bat
    echo [INFO] Da kich hoat moi truong: .env
) else if exist "uv\Scripts\activate.bat" (
    call uv\Scripts\activate.bat
    echo [INFO] Da kich hoat moi truong: uv (tao boi uv)
) else if exist ".uv\Scripts\activate.bat" (
    call .uv\Scripts\activate.bat
    echo [INFO] Da kich hoat moi truong: .uv (tao boi uv)
) else (
    echo [WARNING] Khong tim thay moi truong ao thong dung (.venv, venv, env, .env, uv, .uv). 
    echo Ban co the tu kich hoat bang lenh (vd: conda activate base, hoac su dung 'uv run') truoc khi chay script.
    echo Dang thu chay ngam mac dinh trong PATH...
)

echo Kiem tra moi truong thuc thi...
cmd /c jupyter --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Khong tim thay 'jupyter'. Vui long cai dat jupyter hoac kiem tra PATH.
    pause
    exit /b
)

echo.
echo [1/7] Chay Notebook 00: Data Understanding...
jupyter nbconvert --execute --to notebook --inplace notebooks\00_data_understanding.ipynb
if %errorlevel% neq 0 goto error

echo [2/7] Chay Notebook 01: Exploratory Data Analysis...
jupyter nbconvert --execute --to notebook --inplace notebooks\01_exploratory_data_analysis.ipynb
if %errorlevel% neq 0 goto error

echo [3/7] Chay Notebook 02: Data Cleaning and Preparation...
jupyter nbconvert --execute --to notebook --inplace notebooks\02_data_cleaning_and_preparation.ipynb
if %errorlevel% neq 0 goto error

echo [4/7] Chay Notebook 03: Feature Engineering and Transformation...
jupyter nbconvert --execute --to notebook --inplace notebooks\03_feature_engineering_and_transformation.ipynb
if %errorlevel% neq 0 goto error

echo [5/7] Chay Notebook 04: SVAR Modeling and IRF...
jupyter nbconvert --execute --to notebook --inplace notebooks\04_svar_modeling_and_irf.ipynb
if %errorlevel% neq 0 goto error

echo [6/7] Chay Notebook 05: SARIMA Box Jenkins...
jupyter nbconvert --execute --to notebook --inplace notebooks\05_sarima_box_jenkins.ipynb
if %errorlevel% neq 0 goto error

echo [7/7] Chay Notebook 06: SARIMAX Integration and Evaluation...
jupyter nbconvert --execute --to notebook --inplace notebooks\06_sarimax_integration_and_evaluation.ipynb
if %errorlevel% neq 0 goto error

echo.
echo ==========================================================
echo [SUCCESS] PIPELINE HOAN THANH!
echo Tat ca cac file hinh anh va bang bieu da duoc luu tai outputs/
echo ==========================================================
pause
exit /b

:error
echo.
echo [ERROR] Co loi xay ra trong qua trinh chay pipeline! Vui long kiem tra lai code trong notebook tren.
pause
exit /b
