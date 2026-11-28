param([string]$Deck, [string]$Pdf)
$app = New-Object -ComObject PowerPoint.Application
try {
  $pres = $app.Presentations.Open($Deck, -1, 0, 0)
  try { $pres.SaveAs($Pdf, 32); "saved $Pdf" } finally { $pres.Close() }
} finally { $app.Quit(); [System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null }
