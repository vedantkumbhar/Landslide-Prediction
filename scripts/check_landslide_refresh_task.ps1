[CmdletBinding()]
param(
    [string]$TaskName = "LandslidePrediction_GPM_Refresh",
    [string]$ProjectRoot = "",
    [switch]$RequireRegistered,
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

$RootExists = Test-Path -LiteralPath $ProjectRoot -PathType Container
$ResolvedRoot = if ($RootExists) { (Resolve-Path -LiteralPath $ProjectRoot).Path } else { $ProjectRoot }
$PythonExe = Join-Path $ResolvedRoot ".venv\Scripts\python.exe"
$EntrypointScript = Join-Path $ResolvedRoot "backend\run_scheduled_refresh.py"
$EnvFile = Join-Path $ResolvedRoot ".env"
$ReportDir = Join-Path $ResolvedRoot "outputs\refresh_reports"
$LatestReportPath = Join-Path $ReportDir "latest_refresh_report.json"
$SchedulerLogPath = Join-Path $ReportDir "scheduler.log"
$LockPath = Join-Path $ReportDir ".standard_refresh.lock"

$PythonExists = Test-Path -LiteralPath $PythonExe -PathType Leaf
$EntrypointExists = Test-Path -LiteralPath $EntrypointScript -PathType Leaf
$EnvExists = Test-Path -LiteralPath $EnvFile -PathType Leaf

$CliOk = $false
if ($RootExists -and $PythonExists -and $EntrypointExists) {
    Push-Location $ResolvedRoot
    try {
        & $PythonExe -m backend.run_scheduled_refresh --help *> $null
        $CliOk = ($LASTEXITCODE -eq 0)
    }
    finally {
        Pop-Location
    }
}

$LatestSummary = $null
if (Test-Path -LiteralPath $LatestReportPath -PathType Leaf) {
    try {
        $rawReport = Get-Content -LiteralPath $LatestReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $LatestSummary = [ordered]@{
            run_id                  = $rawReport.run_id
            execution_timestamp_utc = $rawReport.execution_timestamp_utc
            status                  = $rawReport.status
            observation_end_utc     = $rawReport.observation_window.observation_end_utc
        }
    }
    catch {
        $LatestSummary = [ordered]@{ parse_error = $_.Exception.Message }
    }
}

$TaskQueryError = $null
$TaskObj = $null
$TaskInfo = $null
try {
    $TaskObj = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($null -ne $TaskObj) {
        $TaskInfo = Get-ScheduledTaskInfo -TaskName $TaskName -ErrorAction SilentlyContinue
    }
}
catch {
    $TaskQueryError = $_.Exception.Message
}

$TaskConfigValid = $false
$TaskDetails = $null
$ConfigIssues = @()

if ($null -ne $TaskObj) {
    $firstAction = $TaskObj.Actions | Select-Object -First 1
    $firstTrigger = $TaskObj.Triggers | Select-Object -First 1
    $repInterval = if ($null -ne $firstTrigger -and $null -ne $firstTrigger.Repetition) { $firstTrigger.Repetition.Interval } else { $null }
    $startWhenAvail = [bool]$TaskObj.Settings.StartWhenAvailable
    $multiPolicy = [string]$TaskObj.Settings.MultipleInstances

    if ($null -eq $firstAction -or $firstAction.Execute -ne $PythonExe) {
        $ConfigIssues += "Action Execute does not match $PythonExe"
    }
    if ($null -eq $firstAction -or $firstAction.WorkingDirectory -ne $ResolvedRoot) {
        $ConfigIssues += "WorkingDirectory does not match $ResolvedRoot"
    }
    if ($null -eq $firstAction -or $firstAction.Arguments -notmatch 'backend\.run_scheduled_refresh') {
        $ConfigIssues += "Action Arguments do not invoke backend.run_scheduled_refresh"
    }
    if ($repInterval -ne "PT1H") {
        $ConfigIssues += "Expected hourly Repetition.Interval PT1H, got '$repInterval'"
    }
    if (-not $startWhenAvail) {
        $ConfigIssues += "Settings.StartWhenAvailable is False"
    }
    if ($multiPolicy -ne "IgnoreNew") {
        $ConfigIssues += "Settings.MultipleInstances is '$multiPolicy' (expected 'IgnoreNew')"
    }

    $TaskConfigValid = ($ConfigIssues.Count -eq 0)
    $TaskDetails = [ordered]@{
        state                = [string]$TaskObj.State
        execute              = if ($firstAction) { $firstAction.Execute } else { $null }
        arguments            = if ($firstAction) { $firstAction.Arguments } else { $null }
        working_directory    = if ($firstAction) { $firstAction.WorkingDirectory } else { $null }
        repetition_interval  = $repInterval
        start_when_available = $startWhenAvail
        multiple_instances   = $multiPolicy
        last_run_time        = if ($TaskInfo) { [string]$TaskInfo.LastRunTime } else { $null }
        next_run_time        = if ($TaskInfo) { [string]$TaskInfo.NextRunTime } else { $null }
        last_task_result     = if ($TaskInfo) { $TaskInfo.LastTaskResult } else { $null }
        config_valid         = $TaskConfigValid
        config_issues        = $ConfigIssues
    }
}

$Result = [ordered]@{
    task_name             = $TaskName
    registered            = ($null -ne $TaskObj)
    task_query_error      = $TaskQueryError
    project_root          = $ResolvedRoot
    python_executable     = $PythonExe
    entrypoint_script     = $EntrypointScript
    scheduler_log_path    = $SchedulerLogPath
    scheduler_log_exists  = (Test-Path -LiteralPath $SchedulerLogPath -PathType Leaf)
    lock_file_exists      = (Test-Path -LiteralPath $LockPath -PathType Leaf)
    checks                = [ordered]@{
        project_root_ok      = $RootExists
        python_executable_ok = $PythonExists
        entrypoint_script_ok = $EntrypointExists
        env_file_ok          = $EnvExists
        cli_help_check_ok    = $CliOk
    }
    latest_refresh_report = $LatestSummary
    task_details          = $TaskDetails
}

if ($Json) {
    $Result | ConvertTo-Json -Depth 6
}
else {
    Write-Host "========================================================================"
    Write-Host "LANDSLIDE PREDICTION — SCHEDULED TASK STATUS CHECK"
    Write-Host "========================================================================"
    Write-Host "TaskName             : $TaskName"
    Write-Host "Registered           : $($Result.registered)"
    if ($TaskQueryError) {
        Write-Host "TaskQueryNote        : $TaskQueryError"
    }
    Write-Host "ProjectRootOk        : $RootExists ($ResolvedRoot)"
    Write-Host "PythonExecutableOk   : $PythonExists ($PythonExe)"
    Write-Host "EntrypointScriptOk   : $EntrypointExists ($EntrypointScript)"
    Write-Host "EnvFileOk            : $EnvExists"
    Write-Host "CliHelpCheckOk       : $CliOk"
    Write-Host "SchedulerLogExists   : $($Result.scheduler_log_exists) ($SchedulerLogPath)"
    if ($null -ne $TaskDetails) {
        Write-Host "TaskState            : $($TaskDetails.state)"
        Write-Host "RepetitionInterval   : $($TaskDetails.repetition_interval)"
        Write-Host "StartWhenAvailable   : $($TaskDetails.start_when_available)"
        Write-Host "MultipleInstances    : $($TaskDetails.multiple_instances)"
        Write-Host "ConfigValid          : $($TaskDetails.config_valid)"
    }
    Write-Host "========================================================================"
}

if (-not ($RootExists -and $PythonExists -and $EntrypointExists -and $CliOk)) {
    exit 1
}
if ($RequireRegistered -and (-not $Result.registered -or -not $TaskConfigValid)) {
    exit 1
}
exit 0
