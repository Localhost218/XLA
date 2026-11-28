# Dựng báo cáo ngắn AI v2: latexmk -xelatex, chép PDF ra docs/nop-bai/v2/BaoCao_XLA_Nhom10.pdf
# latexmk của MiKTeX cần perl; máy này không cài Strawberry Perl nên mượn perl của Git for Windows.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Get-Command perl -ErrorAction SilentlyContinue)) {
    $gitPerl = 'C:\Program Files\Git\usr\bin'
    if (-not (Test-Path (Join-Path $gitPerl 'perl.exe'))) { throw "không thấy perl (cần cho latexmk)" }
    $env:PATH = "$env:PATH;$gitPerl"   # thêm ở CUỐI để không che lệnh Windows
}
latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
if ($LASTEXITCODE -ne 0) { throw "latexmk lỗi (exit $LASTEXITCODE), xem main.log" }
$outDir = Join-Path $PSScriptRoot '..\..\nop-bai\v2'
New-Item -ItemType Directory -Force $outDir | Out-Null
$out = Join-Path $outDir 'BaoCao_XLA_Nhom10.pdf'
Copy-Item -Force main.pdf $out
Write-Host "Đã ghi $out"
