@echo off
REM Re-run the nine-query demo with the CURRENT registry and replay every trace.
REM Needed because adding the zoom.crop row on 7 Sep changed the registry lock,
REM and a trace recorded under the old lock will not replay under the new one.
REM Waits for queue3 (the zoom nf4 control) to release the card first.
REM Same four adapters as scripts/demo_env.ps1; keep the two in step.
cd /d F:\sih
set PYTHONIOENCODING=utf-8
set SATQUERY_BACKEND=real
set SATQUERY_QWEN=F:\sih\models\Qwen2.5-VL-3B-Instruct
set SATQUERY_VQA_ADAPTER=F:\sih\runs\lora_joint_v3
set SATQUERY_CHANGE_ADAPTER=F:\sih\runs\lora_cdvqa_v4
set SATQUERY_GROUND_ADAPTER=F:\sih\runs\lora_ground_ref_vis
set SATQUERY_FUSION_ADAPTER=F:\sih\runs\lora_fusion_v2
set SATQUERY_VISUAL_TOKENS=256

echo === demo queue registered %DATE% %TIME%, waiting for queue3 to finish > F:\sih\logs\queue_demo.log
:wait
findstr /C:"=== queue3 finished" F:\sih\logs\queue_full3.log >nul 2>&1
if errorlevel 1 (
  timeout /t 120 /nobreak >nul
  goto wait
)
echo === demo run started %DATE% %TIME% >> F:\sih\logs\queue_demo.log
F:\sih\.venv\Scripts\python.exe -u scripts\demo_run.py > F:\sih\logs\demo_run.log 2>&1
echo done %TIME% exit=%ERRORLEVEL% >> F:\sih\logs\queue_demo.log
echo === demo queue finished %DATE% %TIME% >> F:\sih\logs\queue_demo.log
