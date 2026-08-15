<#
.SYNOPSIS
    Register supervise.py as a Windows scheduled task so the bot survives a reboot.

.DESCRIPTION
    supervise.py restarts the bot when the PROCESS dies. It cannot restart the
    MACHINE, so a reboot still ends everything silently. This closes that gap.

    Several defaults in Task Scheduler are actively wrong for a trading bot, and are
    the reason doing this by hand in the GUI usually bites weeks later:

      ExecutionTimeLimit   defaults to 3 DAYS, after which Windows kills the task
                           mid-session. Set to unlimited.
      Battery rules        default to "don't start on battery" and "stop when
                           switching to battery". On a laptop that silently ends
                           trading. Both disabled.
      MultipleInstances    default would let a second copy start alongside the first,
                           two bots on one account fighting over the same ladder.
                           Set to IgnoreNew.
      StartWhenAvailable   so a missed trigger (machine off at the time) still runs.

    Trigger is AT LOGON, not AT STARTUP. At-startup needs the task to run as SYSTEM or
    to store your password; at-logon needs neither, and a bot you cannot see the
    console of is worse than one that waits for you to log in. A 2-minute delay lets
    the network come up first.

.PARAMETER Install
    Actually register the task. Without it this only prints the plan.

.PARAMETER Uninstall
    Remove the task.

.EXAMPLE
    .\install_task.ps1              # show what would be created, change nothing
    .\install_task.ps1 -Install
    .\install_task.ps1 -Uninstall
#>
[CmdletBinding()]
param(
    [switch]$Install,
    [switch]$Uninstall,
    [string]$TaskName = "GridBotSupervisor",
    [int]$DelayMinutes = 2
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path

function Get-PythonPath {
    foreach ($c in @("python", "py")) {
        $cmd = Get-Command $c -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
    }
    throw "No python found on PATH. Install Python or run this from a shell that has it."
}

if ($Uninstall) {
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $existing) { Write-Output "No task named '$TaskName' -- nothing to remove."; exit 0 }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Output "Removed scheduled task '$TaskName'."
    Write-Output "The bot will NOT restart after the next reboot. Anything already"
    Write-Output "running is untouched, and any open position stays on the exchange."
    exit 0
}

$python = Get-PythonPath
$supervisor = Join-Path $repo "supervise.py"
if (-not (Test-Path $supervisor)) { throw "supervise.py not found in $repo" }

Write-Output "Task name        : $TaskName"
Write-Output "Runs             : $python supervise.py"
Write-Output "Working directory: $repo"
Write-Output "Trigger          : at logon of $env:USERNAME, after $DelayMinutes min"
Write-Output "Execution limit  : unlimited (default 3 days would kill it mid-week)"
Write-Output "On battery       : starts and keeps running"
Write-Output "Second instance  : refused"
Write-Output ""

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) { Write-Output "NOTE: a task named '$TaskName' already exists and will be replaced." }

if (-not $Install) {
    Write-Output "This was a dry run. Nothing has been changed."
    Write-Output "Re-run with -Install to register it."
    Write-Output ""
    Write-Output "Before you do, be clear on what it means: after every logon this"
    Write-Output "starts a bot that places real orders on your Binance account with no"
    Write-Output "one watching. supervise.py backs off and trips a breaker after 5"
    Write-Output "crashes in 10 minutes, but nothing stops it trading a bad market."
    exit 0
}

$action = New-ScheduledTaskAction -Execute $python -Argument "supervise.py" -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$trigger.Delay = "PT${DelayMinutes}M"
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Description "Grid bot supervisor (see supervise.py)" -Force | Out-Null

Write-Output "Registered '$TaskName'."
Write-Output ""
Write-Output "It starts at your NEXT logon, not now. To start it immediately:"
Write-Output "    Start-ScheduledTask -TaskName $TaskName"
Write-Output "To stop it and prevent it coming back:"
Write-Output "    Stop-ScheduledTask -TaskName $TaskName; .\install_task.ps1 -Uninstall"
Write-Output ""
Write-Output "Check it is trading with:  py check_run.py"
