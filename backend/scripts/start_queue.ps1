# Register and start the full-split scoring queue as a SYSTEM scheduled task.
# A scheduled task is the only thing on this box that survives the ssh session
# closing - Start-Process does not - and the queue runs for well over a day.
$name = "sih_queue_full"
schtasks /Delete /TN $name /F 2>$null | Out-Null
schtasks /Create /TN $name /TR "F:\sih\queue_full.cmd" /SC ONCE /ST 23:59 /RU SYSTEM /RL HIGHEST /F | Out-Null
schtasks /Run /TN $name | Out-Null
# A ONCE trigger at 23:59 still FIRES at 23:59 if the manual run above has finished by
# then (queue2 re-ran an 85 min split on 7 Sep this way). Disabling after /Run leaves the
# running instance alone and removes the second start.
Start-Sleep -Seconds 3
schtasks /Change /TN $name /DISABLE | Out-Null
"started $name -> F:\sih\logs\queue_full.log"
