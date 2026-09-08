@echo off
REM Second scoring queue for the office 4060: the three overnight adapters
REM (7 Sep) over their whole test splits, nf4, the precision the demo serves.
REM
REM WHY IT WAITS. queue_full.cmd is still running the CDVQA v4 full split on
REM the one card when this is registered. Two evals on an 8 GB card OOM each
REM other, so this polls the first queue's log for its finish line first.
REM
REM ORDER. RSVQA-LR for A (bucket counting) and B (vision tower) first: they
REM are the two candidates to replace joint_v3, 85 min each, and whichever
REM wins is what BEN VQA and the grounding splits get measured against later.
REM CDVQA v5 last: 9.7 h, nothing waits on it tonight.
REM
REM A MUST be asked the bucket way (--count-prompt bucket): it was trained to
REM answer counting with one of the five published labels. The number is
REM comparable because the harness bins both forms to the same five buckets;
REM the summary line records which prompt was used.
cd /d F:\sih
set PYTHONIOENCODING=utf-8

echo === queue2 registered %DATE% %TIME%, waiting for queue_full to finish > F:\sih\logs\queue_full2.log
:wait
findstr /C:"=== queue finished" F:\sih\logs\queue_full.log >nul 2>&1
if errorlevel 1 (
  timeout /t 120 /nobreak >nul
  goto wait
)
echo === queue2 started %DATE% %TIME% >> F:\sih\logs\queue_full2.log

echo [1/3] RSVQA-LR FULL 10,004, joint_cnt, bucket prompt, nf4 - approx 85 min >> F:\sih\logs\queue_full2.log
echo start %TIME% >> F:\sih\logs\queue_full2.log
F:\sih\.venv\Scripts\python.exe -u scripts\eval_rsvqa_lr.py --limit 0 --dtype nf4 --count-prompt bucket --adapter F:\sih\runs\lora_joint_cnt --out F:\sih\runs\rsvqa_lr_full_joint_cnt > F:\sih\logs\rsvqa_full_joint_cnt.log 2>&1
echo done %TIME% exit=%ERRORLEVEL% >> F:\sih\logs\queue_full2.log

echo [2/3] RSVQA-LR FULL 10,004, joint_vis (stopped at 2206 of 3470 steps), nf4 - approx 85 min >> F:\sih\logs\queue_full2.log
echo start %TIME% >> F:\sih\logs\queue_full2.log
F:\sih\.venv\Scripts\python.exe -u scripts\eval_rsvqa_lr.py --limit 0 --dtype nf4 --adapter F:\sih\runs\lora_joint_vis --out F:\sih\runs\rsvqa_lr_full_joint_vis > F:\sih\logs\rsvqa_full_joint_vis.log 2>&1
echo done %TIME% exit=%ERRORLEVEL% >> F:\sih\logs\queue_full2.log

echo [3/3] CDVQA FULL 39,686, cdvqa_v5 answer-balanced, nf4 - approx 9.7 h >> F:\sih\logs\queue_full2.log
echo start %TIME% >> F:\sih\logs\queue_full2.log
F:\sih\.venv\Scripts\python.exe -u scripts\eval_cdvqa.py --limit 0 --dtype nf4 --adapter F:\sih\runs\lora_cdvqa_v5 --out F:\sih\runs\cdvqa_full_nf4_v5 > F:\sih\logs\cdvqa_full_v5.log 2>&1
echo done %TIME% exit=%ERRORLEVEL% >> F:\sih\logs\queue_full2.log

echo === queue2 finished %DATE% %TIME% >> F:\sih\logs\queue_full2.log
