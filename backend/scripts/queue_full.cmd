@echo off
REM Scoring queue for the office 4060. One card, one job at a time.
REM
REM WHY nf4 AND NOT int8. The 400-row probe said int8 was worth +3.2 on
REM RSVQA-LR. The full split says +0.7, because the whole apparent gain was
REM 100 counting questions falling well. int8 costs 35% more decode time, so
REM +0.7 does not buy it: the demo serves nf4 and every number here is quoted
REM in the precision the demo serves. This is the second time a 400-row sample
REM misled us, and the first time it misled a DECISION rather than a number -
REM which is why the int8 probes that were queued here are gone.
REM
REM WHAT IS NOT HERE. Runs A and B are retraining the joint VQA adapter and run
REM C the grounding one, so BEN VQA and BEN grounding get their full splits
REM tomorrow against whichever wins. fusion_v2 and cdvqa_v4 have nothing in
REM flight against them, so their long runs are safe to spend the night on.
REM
REM VRSBench first: it is the third split the statement names, we hold no
REM number on it at all, and four 400-row runs cost under an hour between them.
cd /d F:\sih
set PYTHONIOENCODING=utf-8

echo === queue started %DATE% %TIME% > F:\sih\logs\queue_full.log

echo [1/6] VRSBench VQA 400, base - our first number on this split >> F:\sih\logs\queue_full.log
F:\sih\.venv\Scripts\python.exe -u scripts\eval_vrsbench.py --task vqa --limit 400 --dtype nf4 --out F:\sih\runs\vrsbench_vqa_base_400 > F:\sih\logs\vrs_vqa_base.log 2>&1
echo done %TIME% exit=%ERRORLEVEL% >> F:\sih\logs\queue_full.log

echo [2/6] VRSBench VQA 400, joint_v3 >> F:\sih\logs\queue_full.log
F:\sih\.venv\Scripts\python.exe -u scripts\eval_vrsbench.py --task vqa --limit 400 --dtype nf4 --adapter F:\sih\runs\lora_joint_v3 --out F:\sih\runs\vrsbench_vqa_joint_v3_400 > F:\sih\logs\vrs_vqa_joint.log 2>&1
echo done %TIME% exit=%ERRORLEVEL% >> F:\sih\logs\queue_full.log

echo [3/6] VRSBench referring 400, base >> F:\sih\logs\queue_full.log
F:\sih\.venv\Scripts\python.exe -u scripts\eval_vrsbench.py --task refer --limit 400 --dtype nf4 --out F:\sih\runs\vrsbench_ref_base_400 > F:\sih\logs\vrs_ref_base.log 2>&1
echo done %TIME% exit=%ERRORLEVEL% >> F:\sih\logs\queue_full.log

echo [4/6] VRSBench referring 400, ground_ref_vis - the scale-gap transfer >> F:\sih\logs\queue_full.log
F:\sih\.venv\Scripts\python.exe -u scripts\eval_vrsbench.py --task refer --limit 400 --dtype nf4 --adapter F:\sih\runs\lora_ground_ref_vis --out F:\sih\runs\vrsbench_ref_groundvis_400 > F:\sih\logs\vrs_ref_ground.log 2>&1
echo done %TIME% exit=%ERRORLEVEL% >> F:\sih\logs\queue_full.log

echo [5/6] fusion FULL, 31,386 rows over pairs, nf4 - approx 4.2 h >> F:\sih\logs\queue_full.log
echo start %TIME% >> F:\sih\logs\queue_full.log
F:\sih\.venv\Scripts\python.exe -u scripts\eval_ben_vqa.py --sar --limit 0 --dtype nf4 --adapter F:\sih\runs\lora_fusion_v2 --out F:\sih\runs\fusion_full_nf4_v2 > F:\sih\logs\fusion_full.log 2>&1
echo done %TIME% exit=%ERRORLEVEL% >> F:\sih\logs\queue_full.log

echo [6/6] CDVQA FULL, 39,686 rows, nf4 - approx 9.7 h, preempt this one >> F:\sih\logs\queue_full.log
echo start %TIME% >> F:\sih\logs\queue_full.log
F:\sih\.venv\Scripts\python.exe -u scripts\eval_cdvqa.py --limit 0 --dtype nf4 --adapter F:\sih\runs\lora_cdvqa_v4 --out F:\sih\runs\cdvqa_full_nf4_v4 > F:\sih\logs\cdvqa_full.log 2>&1
echo done %TIME% exit=%ERRORLEVEL% >> F:\sih\logs\queue_full.log

echo === queue finished %DATE% %TIME% >> F:\sih\logs\queue_full.log
