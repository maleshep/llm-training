@echo off
REM Launch Pi (oh-my-pi) coding agent connected to HPC Qwen3.6 inference
REM Make sure the SSH tunnel is running first (start-tunnel.sh)

set PATH=%LOCALAPPDATA%\Programs\omp;%PATH%

echo === Pi (oh-my-pi) + HPC Qwen3.6-35B-A3B ===
echo.
echo Checking tunnel...
curl -s http://localhost:8100/health >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] Tunnel not detected on localhost:8100
    echo        Run: bash start-tunnel.sh
    echo        Or:  ssh -L 8100:demu4xgpu002:8100 -N M316235@onehpc.merckgroup.com
    echo.
    pause
)
echo Tunnel OK - starting Pi...
echo.

omp %*
