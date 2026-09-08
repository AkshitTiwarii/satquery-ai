# What is running on the office box right now, and what the queue logs say.
# Run over ssh: powershell -ExecutionPolicy Bypass -File F:\sih\box_status.ps1
$procs = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(cmd|python|powershell)\.exe$' }
foreach ($p in $procs) {
    $cl = $p.CommandLine
    if ($cl -and $cl.Length -gt 140) { $cl = $cl.Substring(0, 140) }
    "{0,6} {1,6} {2} {3}" -f $p.ProcessId, $p.ParentProcessId, $p.CreationDate.ToString("HH:mm:ss"), $cl
}
"--- tasks"
Get-ScheduledTask -TaskName sih_queue* | ForEach-Object { "{0,-20} {1}" -f $_.TaskName, $_.State }
foreach ($f in @("queue_full2.log", "queue_full3.log", "queue_demo.log")) {
    "--- $f"
    if (Test-Path -LiteralPath "F:\sih\logs\$f") { Get-Content -LiteralPath "F:\sih\logs\$f" }
}
