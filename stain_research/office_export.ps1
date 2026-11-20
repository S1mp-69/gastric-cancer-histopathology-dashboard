param([string]$InputDocx,
      [string]$InputPptx,
      [string]$PaginationJson,
      [Parameter(Mandatory=$true)][string]$OutputPdf)
$ErrorActionPreference = 'Stop'
$wordApplication = $null
$wordDocument = $null
$powerpointApplication = $null
$powerpointPresentation = $null
[object]$wordSaveChanges = 0
try {
    if ($InputDocx) {
        $wordApplication = New-Object -ComObject Word.Application
        $wordApplication.Visible = $false
        $wordApplication.DisplayAlerts = 0
        $wordDocument = $wordApplication.Documents.Open($InputDocx, $false, $true, $false)
        $wordDocument.Fields.Update() | Out-Null
        $wordDocument.Repaginate()
        $wordDocument.Fields.Update() | Out-Null
        if ($PaginationJson) {
            $reportPageMap = @{}
            foreach ($reportBookmark in $wordDocument.Bookmarks) {
                $reportPageMap[$reportBookmark.Name] = $reportBookmark.Range.Information(1)
            }
            $reportPageMap | ConvertTo-Json | Set-Content -LiteralPath $PaginationJson -Encoding UTF8
        }
        $wordDocument.ExportAsFixedFormat($OutputPdf, 17)
        Write-Output "Native Word PDF export completed: $OutputPdf"
    } elseif ($InputPptx) {
        $powerpointApplication = New-Object -ComObject PowerPoint.Application
        $powerpointApplication.DisplayAlerts = 1
        $powerpointPresentation = $powerpointApplication.Presentations.Open($InputPptx, $true, $false, $false)
        $powerpointPresentation.SaveAs($OutputPdf, 32)
        Write-Output "Native PowerPoint PDF export completed: $OutputPdf"
    } else { throw 'InputDocx or InputPptx is required.' }
} finally {
    if ($null -ne $wordDocument) { $wordDocument.Close([ref]$wordSaveChanges) }
    if ($null -ne $wordApplication) { $wordApplication.Quit([ref]$wordSaveChanges) }
    if ($null -ne $powerpointPresentation) { $powerpointPresentation.Close() }
    if ($null -ne $powerpointApplication) { $powerpointApplication.Quit() }
    if ($null -ne $wordDocument) { [System.Runtime.InteropServices.Marshal]::ReleaseComObject($wordDocument) | Out-Null }
    if ($null -ne $wordApplication) { [System.Runtime.InteropServices.Marshal]::ReleaseComObject($wordApplication) | Out-Null }
    if ($null -ne $powerpointPresentation) { [System.Runtime.InteropServices.Marshal]::ReleaseComObject($powerpointPresentation) | Out-Null }
    if ($null -ne $powerpointApplication) { [System.Runtime.InteropServices.Marshal]::ReleaseComObject($powerpointApplication) | Out-Null }
}
exit 0
