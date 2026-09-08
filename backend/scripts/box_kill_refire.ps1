# A ONCE task registered with /ST 23:59 fires at 23:59 IF its manual run has already
# finished - queue2 finished 22:47 and re-fired at 23:59 on 7 Sep, re-running an 85 min
# full split against the card queue3 was using. Ends that instance and disables every
# sih_queue* trigger so none of them can fire again. Running instances are not touched
# by /DISABLE; only the re-fired queue2 is ended.
schtasks /End /TN sih_queue_full2 | Out-Null
Start-Sleep -Seconds 2
$stale = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and ($_.CommandLine -match 'queue_full2.cmd' -or $_.CommandLine -match 'rsvqa_lr_full_joint_cnt')
}
foreach ($p in $stale) { "killing {0} {1}" -f $p.ProcessId, $p.CommandLine.Substring(0, [Math]::Min(90, $p.CommandLine.Length)); Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
foreach ($t in @("sih_queue_full", "sih_queue_full2", "sih_queue_full3", "sih_queue_demo")) {
    schtasks /Change /TN $t /DISABLE | Out-Null
}
"--- after"
Get-ScheduledTask -TaskName sih_queue* | ForEach-Object { "{0,-20} {1}" -f $_.TaskName, $_.State }
Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(cmd|python)\.exe$' } | ForEach-Object {
    $cl = $_.CommandLine; if ($cl -and $cl.Length -gt 110) { $cl = $cl.Substring(0, 110) }
    "{0,6} {1} {2}" -f $_.ProcessId, $_.CreationDate.ToString("HH:mm:ss"), $cl
}
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
