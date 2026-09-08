# Register and start the demo re-run as a SYSTEM task; the .cmd waits for queue3.
$name = "sih_queue_demo"
schtasks /Delete /TN $name /F 2>$null | Out-Null
schtasks /Create /TN $name /TR "F:\sih\queue_demo.cmd" /SC ONCE /ST 23:59 /RU SYSTEM /RL HIGHEST /F | Out-Null
schtasks /Run /TN $name | Out-Null
# A ONCE trigger at 23:59 still FIRES at 23:59 if the manual run above has finished by
# then (queue2 re-ran an 85 min split on 7 Sep this way). Disabling after /Run leaves the
# running instance alone and removes the second start.
Start-Sleep -Seconds 3
schtasks /Change /TN $name /DISABLE | Out-Null
"started $name -> F:\sih\logs\queue_demo.log"
