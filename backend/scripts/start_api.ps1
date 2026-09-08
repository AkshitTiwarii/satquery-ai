# Run the SatQuery HTTP API on the office box as a SYSTEM scheduled task, so it survives
# the ssh session closing (Start-Process does not on this box).
#
#   powershell -ExecutionPolicy Bypass -File F:\sih\start_api.ps1 -Backend real   # the demo
#   powershell -ExecutionPolicy Bypass -File F:\sih\start_api.ps1 -Backend stub   # connectivity only
#   powershell -ExecutionPolicy Bypass -File F:\sih\start_api.ps1 -Stop
#
# Env is the same four adapters as scripts/demo_env.ps1, written into a wrapper .cmd because
# a scheduled task carries no environment of its own. Listens on 0.0.0.0:8765; the firewall
# rule is added once (Tailscale-only traffic reaches this box anyway).
param(
    [ValidateSet("real", "stub")] [string] $Backend = "real",
    [int] $Port = 8765,
    [string] $Token = "",
    [switch] $Stop
)
$name = "sih_api"
schtasks /End /TN $name 2>$null | Out-Null
schtasks /Delete /TN $name /F 2>$null | Out-Null
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine -match "satquery.api" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
if ($Stop) { "stopped $name"; exit 0 }

$cmd = @"
@echo off
cd /d F:\sih
set PYTHONIOENCODING=utf-8
set SATQUERY_BACKEND=$Backend
set SATQUERY_QWEN=F:\sih\models\Qwen2.5-VL-3B-Instruct
set SATQUERY_VQA_ADAPTER=F:\sih\runs\lora_joint_v3
set SATQUERY_CHANGE_ADAPTER=F:\sih\runs\lora_cdvqa_v4
set SATQUERY_GROUND_ADAPTER=F:\sih\runs\lora_ground_ref_vis
set SATQUERY_FUSION_ADAPTER=F:\sih\runs\lora_fusion_v2
set SATQUERY_VISUAL_TOKENS=256
set SATQUERY_API_TOKEN=$Token
F:\sih\.venv\Scripts\python.exe -u -m satquery.api --host 0.0.0.0 --port $Port > F:\sih\logs\api.log 2>&1
"@
Set-Content -LiteralPath "F:\sih\api_run.cmd" -Value $cmd -Encoding ASCII

if (-not (Get-NetFirewallRule -DisplayName "SatQuery API $Port" -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName "SatQuery API $Port" -Direction Inbound -Protocol TCP -LocalPort $Port -Action Allow | Out-Null
    "firewall rule added for tcp/$Port"
}
schtasks /Create /TN $name /TR "F:\sih\api_run.cmd" /SC ONCE /ST 23:59 /RU SYSTEM /RL HIGHEST /F | Out-Null
schtasks /Run /TN $name | Out-Null
Start-Sleep -Seconds 3
schtasks /Change /TN $name /DISABLE | Out-Null
"started $name backend=$Backend on 0.0.0.0:$Port -> F:\sih\logs\api.log"
