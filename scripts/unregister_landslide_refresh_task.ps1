[CmdletBinding()]
param(
    [string]$TaskName = "LandslidePrediction_GPM_Refresh",
    [switch]$DryRun,
    [switch]$Json
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Safety: reject wildcards, paths, or blank names so only this named task can ever be targeted.
if ([string]::IsNullOrWhiteSpace($TaskName) -or $TaskName -match '[\*\?\\/]') {
    throw "Invalid TaskName '$TaskName'. Provide an exact non-empty task name without wildcards or path separators."
}

$ExistingTask = $null
$TaskQueryError = $null
try {
    $ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
}
catch {
    $TaskQueryError = $_.Exception.Message
}
$Found = ($null -ne $ExistingTask)

if ($DryRun) {
    $Summary = [ordered]@{
        task_name             = $TaskName
        dry_run               = $true
        task_found            = $Found
        task_query_error      = $TaskQueryError
        action_taken          = $(if ($TaskQueryError) { "DRY_RUN_CIM_UNAVAILABLE" } elseif ($Found) { "WOULD_UNREGISTER" } else { "NO_OP_NOT_REGISTERED" })
        project_data_touched  = $false
        logs_or_backups_kept  = $true
    }
    if ($Json) {
        $Summary | ConvertTo-Json -Depth 4
    }
    else {
        Write-Host "========================================================================"
        Write-Host "LANDSLIDE PREDICTION — UNREGISTER SCHEDULED TASK (DRY RUN)"
        Write-Host "========================================================================"
        Write-Host "TaskName            : $TaskName"
        Write-Host "TaskFound           : $Found"
        if ($TaskQueryError) {
            Write-Host "TaskQueryNote       : $TaskQueryError"
        }
        Write-Host "PlannedAction       : $($Summary.action_taken)"
        Write-Host "PreservesData/Logs  : True (project data, reports, logs, and backups untouched)"
        Write-Host "========================================================================"
    }
    exit 0
}

if ($TaskQueryError) {
    throw "Unable to query Windows Scheduled Tasks: $TaskQueryError"
}

if (-not $Found) {
    if ($Json) {
        [ordered]@{
            task_name            = $TaskName
            unregistered         = $false
            reason               = "Task not registered"
            project_data_touched = $false
        } | ConvertTo-Json -Depth 3
    }
    else {
        Write-Host "Scheduled task '$TaskName' is not registered; nothing removed. Project data, reports, logs, and backups untouched."
    }
    exit 0
}

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false

if ($Json) {
    [ordered]@{
        task_name            = $TaskName
        unregistered         = $true
        project_data_touched = $false
    } | ConvertTo-Json -Depth 3
}
else {
    Write-Host "Unregistered scheduled task '$TaskName'. Project data, reports, logs, and backups were preserved untouched."
}
