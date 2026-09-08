# The demo environment on the office box: one resident Qwen, four adapters.
#   . F:\sih\demo_env.ps1        (dot-source, then `python -m satquery.run ...`)
# Every value here is a measured choice - PLAN section 9 has the number behind each.
$env:SATQUERY_BACKEND = "real"
$env:SATQUERY_QWEN = "F:\sih\models\Qwen2.5-VL-3B-Instruct"     # local weights; the hub id would pull 7.5 GB
$env:SATQUERY_VQA_ADAPTER = "F:\sih\runs\lora_joint_v3"         # RSVQA-LR 84.5 / BigEarthNet 78.0 (5 Sep)
$env:SATQUERY_CHANGE_ADAPTER = "F:\sih\runs\lora_cdvqa_v4"      # CDVQA 58.5 vs 40.0 base
$env:SATQUERY_GROUND_ADAPTER = "F:\sih\runs\lora_ground_ref_vis" # point 87.0% Acc@0.5, reference 30.0% (vision-tower LoRA, 6 Sep)
$env:SATQUERY_FUSION_ADAPTER = "F:\sih\runs\lora_fusion_v2"     # 79.8 over optical+SAR, 73.0 radar-only
$env:SATQUERY_VISUAL_TOKENS = "256"
$env:PYTHONIOENCODING = "utf-8"
Set-Location F:\sih
"satquery demo env: 4 adapters, backend=real"
