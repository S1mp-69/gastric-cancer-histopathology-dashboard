# Resume only the bounded publication queue; mutex prevents duplicate launches.
$ErrorActionPreference = 'Stop'
$publicationRoot = Split-Path -Parent $PSScriptRoot
$publicationMutex = New-Object System.Threading.Mutex($false, 'Gastric_HiESD_WithinClass_V1')
if (-not $publicationMutex.WaitOne(0)) { Write-Output 'The publication workflow is already running.'; exit 2 }
Add-Type -TypeDefinition @'
using System.Runtime.InteropServices;
public class GastricPublicationAwake { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags); }
'@
try {
    [GastricPublicationAwake]::SetThreadExecutionState([uint32]2147483649) | Out-Null
    $publicationPython = Join-Path $publicationRoot '.venv-research/Scripts/python.exe'
    $publicationLogs = Join-Path $publicationRoot '.sync-backups/hiesd_progress'
    New-Item -ItemType Directory -Path $publicationLogs -Force | Out-Null
    $publicationQueue = Start-Process -FilePath $publicationPython -ArgumentList @('-u','stain_research/within_class_publication_queue.py') -WorkingDirectory $publicationRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $publicationLogs 'within-publication.log') -RedirectStandardError (Join-Path $publicationLogs 'within-publication-error.log') -PassThru
    $publicationQueue.WaitForExit()
    if ($publicationQueue.ExitCode -ne 0) { throw "Publication workflow exited $($publicationQueue.ExitCode). Read saved FAILURE.json and logs." }
} finally {
    [GastricPublicationAwake]::SetThreadExecutionState([uint32]2147483648) | Out-Null
    $publicationMutex.ReleaseMutex();$publicationMutex.Dispose()
}
