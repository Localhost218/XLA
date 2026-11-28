param([string]$Deck, [string]$OutDir, [string]$Slides)
$app = New-Object -ComObject PowerPoint.Application
try {
  # Open(FileName, ReadOnly=-1, Untitled=0, WithWindow=0)
  $pres = $app.Presentations.Open($Deck, -1, 0, 0)
  try {
    "count=" + $pres.Slides.Count
    foreach ($i in ($Slides -split "[,; ]+" | ForEach-Object { [int]$_ })) {
      $out = Join-Path $OutDir ("slide-{0:D2}.png" -f $i)
      $pres.Slides.Item($i).Export($out, "PNG", 1600, 900)
      "exported $out"
    }
  } finally { $pres.Close() }
} finally {
  $app.Quit()
  [System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null
}
