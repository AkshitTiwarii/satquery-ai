@echo off
REM Third scoring queue on the office 4060: the zoom-and-re-ask step (rule Z1),
REM measured on VRSBench, the one prescribed split where it can apply (512 px,
REM sub-metre). Waits for queue_full2 (cdvqa_v5 full split) to finish first -
REM two evals on the one card OOM each other.
REM
REM Four 400-row runs, same rows as the 6 Sep numbers (base 43.4 VQA / 37.8
REM referring; adapter 42.7 / 13.0), each WITH --zoom. The gain is the delta to
REM those. 400 rows is enough to tell +10 from 0; the full split follows only
REM if the sign is right. Under an hour and a half in total.
cd /d F:\sih
set PYTHONIOENCODING=utf-8

echo === queue3 registered %DATE% %TIME%, waiting for queue_full2 to finish > F:\sih\logs\queue_full3.log
:wait
findstr /C:"=== queue2 finished" F:\sih\logs\queue_full2.log >nul 2>&1
if errorlevel 1 (
  timeout /t 120 /nobreak >nul
  goto wait
)
echo === queue3 started %DATE% %TIME% >> F:\sih\logs\queue_full3.log

echo [1/4] VRSBench referring 400, base + zoom (vs 37.8) >> F:\sih\logs\queue_full3.log
F:\sih\.venv\Scripts\python.exe -u scripts\eval_vrsbench.py --task refer --limit 400 --dtype nf4 --zoom --out F:\sih\runs\vrsbench_ref_base_zoom_400 > F:\sih\logs\vrs_ref_base_zoom.log 2>&1
echo done %TIME% exit=%ERRORLEVEL% >> F:\sih\logs\queue_full3.log

echo [2/4] VRSBench referring 400, ground_ref_vis + zoom (vs 13.0) >> F:\sih\logs\queue_full3.log
F:\sih\.venv\Scripts\python.exe -u scripts\eval_vrsbench.py --task refer --limit 400 --dtype nf4 --zoom --adapter F:\sih\runs\lora_ground_ref_vis --out F:\sih\runs\vrsbench_ref_groundvis_zoom_400 > F:\sih\logs\vrs_ref_ground_zoom.log 2>&1
echo done %TIME% exit=%ERRORLEVEL% >> F:\sih\logs\queue_full3.log

echo [3/4] VRSBench VQA 400, base + zoom (vs 43.4) >> F:\sih\logs\queue_full3.log
F:\sih\.venv\Scripts\python.exe -u scripts\eval_vrsbench.py --task vqa --limit 400 --dtype nf4 --zoom --out F:\sih\runs\vrsbench_vqa_base_zoom_400 > F:\sih\logs\vrs_vqa_base_zoom.log 2>&1
echo done %TIME% exit=%ERRORLEVEL% >> F:\sih\logs\queue_full3.log

echo [4/4] VRSBench VQA 400, joint_v3 + zoom (vs 42.7) >> F:\sih\logs\queue_full3.log
F:\sih\.venv\Scripts\python.exe -u scripts\eval_vrsbench.py --task vqa --limit 400 --dtype nf4 --zoom --adapter F:\sih\runs\lora_joint_v3 --out F:\sih\runs\vrsbench_vqa_joint_v3_zoom_400 > F:\sih\logs\vrs_vqa_joint_zoom.log 2>&1
echo done %TIME% exit=%ERRORLEVEL% >> F:\sih\logs\queue_full3.log

echo === queue3 finished %DATE% %TIME% >> F:\sih\logs\queue_full3.log
