param([string]$Pptx, [string]$OutDir, [int]$Width = 1600)
# Xuat moi slide ra PNG bang PowerPoint COM (khong mo cua so).
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force $OutDir | Out-Null
Get-ChildItem $OutDir -Filter 'slide-*.png' | Remove-Item -Force
$app = New-Object -ComObject PowerPoint.Application
try {
  $pres = $app.Presentations.Open($Pptx, $true, $false, $false)  # ReadOnly, Untitled=false, WithWindow=false
  $h = [int]($Width * $pres.PageSetup.SlideHeight / $pres.PageSetup.SlideWidth)
  $n = $pres.Slides.Count
  for ($i = 1; $i -le $n; $i++) {
    $f = Join-Path $OutDir ('slide-{0:D2}.png' -f $i)
    $pres.Slides.Item($i).Export($f, 'PNG', $Width, $h)
  }
  $pres.Close()
  "rendered $n slides to $OutDir"
} finally {
  $app.Quit()
  [System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null
}
