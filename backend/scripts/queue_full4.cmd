@echo off
REM Fourth scoring queue: RAO's rank-16 grounding adapter (7 Sep, kernel
REM ayushrewd/notebook82c03edf4b: in-kernel fp16 reference 31.0 / point 88.0 vs
REM rank-8's 33.0 / 86.5 on the same 400 rows). Waits for the demo re-run to
REM release the card.
REM
REM [1] is the decision-grade probe in the demo's precision (compare 30.0 / 87.0).
REM [2] and [3] are the FULL BEN box test split, 15,104 rows, for the live adapter
REM and the challenger - grounding has never had a full-split number and the rule
REM is rank on the full split or not at all. ~10-12 h each at grounding decode
REM lengths; the card is idle overnight anyway. Kill [3] if Tuesday needs the card
REM for the team's RSVQA full splits: the demo box must be free by Tue evening.
cd /d F:\sih
set PYTHONIOENCODING=utf-8

echo === queue4 registered %DATE% %TIME%, waiting for the demo re-run to finish > F:\sih\logs\queue_full4.log
:wait
findstr /C:"=== demo queue finished" F:\sih\logs\queue_demo.log >nul 2>&1
if errorlevel 1 (
  timeout /t 120 /nobreak >nul
  goto wait
)
echo === queue4 started %DATE% %TIME% >> F:\sih\logs\queue_full4.log

echo [1/3] grounding 400, rank 16, nf4 (vs 30.0 ref / 87.0 point) >> F:\sih\logs\queue_full4.log
F:\sih\.venv\Scripts\python.exe -u scripts\eval_ground.py --limit 400 --dtype nf4 --adapter F:\sih\runs\lora_ground_r16 --out F:\sih\runs\ground_lora_r16_400 > F:\sih\logs\ground_r16_400.log 2>&1
echo done %TIME% exit=%ERRORLEVEL% >> F:\sih\logs\queue_full4.log

echo [2/3] grounding FULL 15,104, ground_ref_vis (live), nf4 >> F:\sih\logs\queue_full4.log
echo start %TIME% >> F:\sih\logs\queue_full4.log
F:\sih\.venv\Scripts\python.exe -u scripts\eval_ground.py --limit 0 --dtype nf4 --adapter F:\sih\runs\lora_ground_ref_vis --out F:\sih\runs\ground_lora_ref_vis_full > F:\sih\logs\ground_ref_vis_full.log 2>&1
echo done %TIME% exit=%ERRORLEVEL% >> F:\sih\logs\queue_full4.log

echo [3/3] grounding FULL 15,104, rank 16, nf4 >> F:\sih\logs\queue_full4.log
echo start %TIME% >> F:\sih\logs\queue_full4.log
F:\sih\.venv\Scripts\python.exe -u scripts\eval_ground.py --limit 0 --dtype nf4 --adapter F:\sih\runs\lora_ground_r16 --out F:\sih\runs\ground_lora_r16_full > F:\sih\logs\ground_r16_full.log 2>&1
echo done %TIME% exit=%ERRORLEVEL% >> F:\sih\logs\queue_full4.log

echo === queue4 finished %DATE% %TIME% >> F:\sih\logs\queue_full4.log
