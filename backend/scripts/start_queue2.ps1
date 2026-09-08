# Register and start the second full-split scoring queue as a SYSTEM task.
# Same shape as start_queue.ps1: a scheduled task is the only thing on this
# box that survives the ssh session closing. The .cmd itself waits for
# queue_full to finish before touching the card.
$name = "sih_queue_full2"
schtasks /Delete /TN $name /F 2>$null | Out-Null
schtasks /Create /TN $name /TR "F:\sih\queue_full2.cmd" /SC ONCE /ST 23:59 /RU SYSTEM /RL HIGHEST /F | Out-Null
schtasks /Run /TN $name | Out-Null
# A ONCE trigger at 23:59 still FIRES at 23:59 if the manual run above has finished by
# then (queue2 re-ran an 85 min split on 7 Sep this way). Disabling after /Run leaves the
# running instance alone and removes the second start.
Start-Sleep -Seconds 3
schtasks /Change /TN $name /DISABLE | Out-Null
"started $name -> F:\sih\logs\queue_full2.log"
