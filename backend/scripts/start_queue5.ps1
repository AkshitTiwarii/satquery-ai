# Register and start the fifth scoring queue (Khushi's ref40k grounding probe) as a SYSTEM task.
# The .cmd waits for queue4 to finish before it touches the card.
$name = "sih_queue_full5"
schtasks /Delete /TN $name /F 2>$null | Out-Null
schtasks /Create /TN $name /TR "F:\sih\queue_full5.cmd" /SC ONCE /ST 23:59 /RU SYSTEM /RL HIGHEST /F | Out-Null
schtasks /Run /TN $name | Out-Null
# A ONCE trigger at 23:59 still FIRES at 23:59 if the manual run above has finished by
# then. Disabling after /Run leaves the running instance alone and removes the second start.
Start-Sleep -Seconds 3
schtasks /Change /TN $name /DISABLE | Out-Null
"started $name -> F:\sih\logs\queue_full5.log"
