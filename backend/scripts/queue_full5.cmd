@echo off
REM Fifth scoring queue: Khushi's "40k reference" grounding adapter (8 Sep, kernel
REM khushiijainn32/notebook67279e1276). The generator turned --ref-rows 40000 into a
REM per-patch cap, so it trained on 10,986 reference rows - the rank-8 recipe re-run with a
REM new shuffle (in-kernel fp16 400: reference 32.0 / point 88.5 vs 33.0 / 86.5). This nf4
REM probe is the seed-noise floor for the 400-row grounding probe in the demo's precision.
REM Waits for queue4 to release the card.
cd /d F:\sih
set PYTHONIOENCODING=utf-8

echo === queue5 registered %DATE% %TIME%, waiting for queue4 to finish > F:\sih\logs\queue_full5.log
:wait
findstr /C:"=== queue4 finished" F:\sih\logs\queue_full4.log >nul 2>&1
if errorlevel 1 (
  timeout /t 120 /nobreak >nul
  goto wait
)
echo === queue5 started %DATE% %TIME% >> F:\sih\logs\queue_full5.log

echo [1/1] grounding 400, ref40k (really 10,986 ref rows), nf4 (vs 30.0 ref / 87.0 point) >> F:\sih\logs\queue_full5.log
F:\sih\.venv\Scripts\python.exe -u scripts\eval_ground.py --limit 400 --dtype nf4 --adapter F:\sih\runs\lora_ground_ref40k --out F:\sih\runs\ground_lora_ref40k_400 > F:\sih\logs\ground_ref40k_400.log 2>&1
echo done %TIME% exit=%ERRORLEVEL% >> F:\sih\logs\queue_full5.log

echo === queue5 finished %DATE% %TIME% >> F:\sih\logs\queue_full5.log
