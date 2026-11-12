[CmdletBinding()]
param(
    [string]$TaskName = "LandslidePrediction_GPM_Refresh",
    [string]$ProjectRoot = "",
    [Nullable[datetime]]$StartAt = $null,
    [double]$MaxLatencyHours = 12.0,
    [int]$Workers = 4,
    [switch]$DryRun,
    [switch]$Force,
    [switch]$Json
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($TaskName) -or $TaskName -match '[\*\?\\/]') {
    throw "Invalid TaskName '$TaskName'. Provide an exact non-empty task name without wildcards or path separators."
}

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}
$ResolvedRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$PythonExe = Join-Path $ResolvedRoot ".venv\Scripts\python.exe"
$EntrypointScript = Join-Path $ResolvedRoot "backend\run_scheduled_refresh.py"
$EnvFile = Join-Path $ResolvedRoot ".env"
$ReportDir = Join-Path $ResolvedRoot "outputs\refresh_reports"
$SchedulerLog = Join-Path $ReportDir "scheduler.log"

if (-not (Test-Path -LiteralPath $ResolvedRoot -PathType Container)) {
    throw "Project directory not found: $ResolvedRoot"
}
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Virtualenv Python executable not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $EntrypointScript -PathType Leaf)) {
    throw "Scheduled refresh entrypoint not found: $EntrypointScript"
}
if ($MaxLatencyHours -le 0 -or $MaxLatencyHours -gt 12.0) {
    throw "MaxLatencyHours must be within (0, 12.0]; got $MaxLatencyHours. The 12-hour freshness guard cannot be weakened."
}
if ($Workers -lt 1 -or $Workers -gt 16) {
    throw "Workers must be between 1 and 16; got $Workers."
}

# Verify the scheduled entrypoint imports and parses CLI options cleanly without executing a refresh
Push-Location $ResolvedRoot
try {
    & $PythonExe -m backend.run_scheduled_refresh --help *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Entrypoint CLI self-check failed ($PythonExe -m backend.run_scheduled_refresh --help returned exit code $LASTEXITCODE)."
    }
}
finally {
    Pop-Location
}

$LatencyFormatted = [string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0:0.0}", $MaxLatencyHours)
$TaskArguments = "-m backend.run_scheduled_refresh --max-latency-hours $LatencyFormatted --workers $Workers"

if ($null -eq $StartAt) {
    $nowLocal = Get-Date
    $nextHour = $nowLocal.Date.AddHours($nowLocal.Hour + 1)
    $EffectiveStartAt = $nextHour
}
else {
    $EffectiveStartAt = $StartAt.Value
}

$UserId = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

$Config = [ordered]@{
    task_name                 = $TaskName
    dry_run                   = [bool]$DryRun
    project_root              = $ResolvedRoot
    python_executable         = $PythonExe
    entrypoint_module         = "backend.run_scheduled_refresh"
    entrypoint_script         = $EntrypointScript
    task_arguments            = $TaskArguments
    working_directory         = $ResolvedRoot
    scheduler_log             = $SchedulerLog
    env_file                  = $EnvFile
    env_file_exists           = (Test-Path -LiteralPath $EnvFile -PathType Leaf)
    report_dir_exists         = (Test-Path -LiteralPath $ReportDir -PathType Container)
    max_latency_hours         = [double]$MaxLatencyHours
    unchanged_window_skip     = $true
    recurrence                = "Hourly (every 1 hour, PT1H, indefinite repetition on -Once trigger)"
    repetition_interval       = "01:00:00"
    start_at_local            = $EffectiveStartAt.ToString("yyyy-MM-ddTHH:mm:ssK")
    start_when_available      = $true
    multiple_instances_policy = "IgnoreNew"
    execution_time_limit      = "02:00:00"
    wake_to_run               = $false
    run_while_shut_down       = $false
    principal_user            = $UserId
    principal_logon_type      = "Interactive"
    principal_run_level       = "Limited"
    prerequisites_note        = "The PC must be powered on (logged-in session for Interactive logon) and connected to the internet for live NASA CMR discovery/granule retrieval. Eligible missed runs start automatically once powered on and available."
    checks                    = [ordered]@{
        project_root_ok       = $true
        python_executable_ok  = $true
        entrypoint_script_ok  = $true
        cli_help_check_ok     = $true
        freshness_guard_ok    = $true
    }
}

if ($DryRun) {
    if ($Json) {
        $Config | ConvertTo-Json -Depth 5
    }
    else {
        Write-Host "========================================================================"
        Write-Host "LANDSLIDE PREDICTION — WINDOWS TASK SCHEDULER REGISTRATION (DRY RUN)"
        Write-Host "========================================================================"
        Write-Host "No scheduled task was registered, modified, or executed."
        Write-Host ""
        $Config.GetEnumerator() | ForEach-Object {
            if ($_.Key -ne "checks") {
                Write-Host ("{0,-28}: {1}" -f $_.Key, $_.Value)
            }
        }
        Write-Host ""
        Write-Host "Validation Checks:"
        $Config.checks.GetEnumerator() | ForEach-Object {
            Write-Host ("  - {0,-24}: {1}" -f $_.Key, $_.Value)
        }
        Write-Host ""
        Write-Host "Equivalent Windows Scheduled Task PowerShell Specification:"
        Write-Host ("  Action    : New-ScheduledTaskAction -Execute `"{0}`" -Argument `"{1}`" -WorkingDirectory `"{2}`"" -f $PythonExe, $TaskArguments, $ResolvedRoot)
        Write-Host ("  Trigger   : New-ScheduledTaskTrigger -Once -At `"{0}`" -RepetitionInterval (New-TimeSpan -Hours 1)" -f $EffectiveStartAt.ToString("yyyy-MM-dd HH:mm:ss"))
        Write-Host "  Settings  : New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 2) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries"
        Write-Host ("  Principal : New-ScheduledTaskPrincipal -UserId `"{0}`" -LogonType Interactive -RunLevel Limited" -f $UserId)
        Write-Host "========================================================================"
    }
    exit 0
}

$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -ne $ExistingTask -and -not $Force) {
    throw "Scheduled task '$TaskName' already exists. Re-run with -Force to replace it, or unregister it first."
}

$Action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument $TaskArguments `
    -WorkingDirectory $ResolvedRoot

$Trigger = New-ScheduledTaskTrigger `
    -Once `
    -At $EffectiveStartAt `
    -RepetitionInterval (New-TimeSpan -Hours 1)

$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

$Principal = New-ScheduledTaskPrincipal `
    -UserId $UserId `
    -LogonType Interactive `
    -RunLevel Limited

$Description = (
    "Hourly NASA GPM IMERG V07C landslide hazard refresh check via backend.run_scheduled_refresh. " +
    "Enforces 12h freshness guard, unchanged-window skipping, process lock, and atomic HTML outputs. " +
    "Requires PC powered on with internet access; logs to outputs/refresh_reports/scheduler.log."
)

$RegisterParams = @{
    TaskName    = $TaskName
    Action      = $Action
    Trigger     = $Trigger
    Settings    = $Settings
    Principal   = $Principal
    Description = $Description
}
if ($Force) {
    $RegisterParams["Force"] = $true
}

Register-ScheduledTask @RegisterParams | Out-Null
Write-Host "Registered scheduled task '$TaskName' (hourly repetition starting at $($EffectiveStartAt.ToString('yyyy-MM-dd HH:mm:ss')))."
