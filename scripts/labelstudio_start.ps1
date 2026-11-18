# Khởi động Label Studio local cho phase 08 (project xla-test-v1). Chỉ nghe 127.0.0.1.
# Dùng:  powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\labelstudio_start.ps1 [-Port 8080] [-Stop]
# Dữ liệu (DB, log, credentials.json) ở D:\DevCache\label-studio; ảnh phục vụ từ <repo>\data\real\frames.
# Server chạy tách khỏi cửa sổ gọi (Start-Process), log ở D:\DevCache\label-studio\logs\.
param(
    [int]$Port = 8080,
    [string]$DataDir = 'D:\DevCache\label-studio',
    [string]$Exe = 'D:\DevCache\uv-tools\bin\label-studio.exe',
    [string]$DocRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'data\real'),
    [switch]$Stop
)
$ErrorActionPreference = 'Stop'

$running = Get-CimInstance Win32_Process -Filter "Name = 'label-studio.exe'" -ErrorAction SilentlyContinue
if ($Stop) {
    $running | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    Write-Output "đã dừng $(@($running).Count) tiến trình label-studio"
    exit 0
}
if ($running) { Write-Output "label-studio đang chạy (pid $(@($running.ProcessId) -join ','))"; exit 0 }

$credPath = Join-Path $DataDir 'credentials.json'
if (-not (Test-Path $credPath)) { throw "thiếu $credPath (tài khoản + token, tạo lúc dựng server lần đầu)" }
$cred = Get-Content $credPath -Raw | ConvertFrom-Json

New-Item -ItemType Directory -Force -Path (Join-Path $DataDir 'logs') | Out-Null
$env:LABEL_STUDIO_BASE_DATA_DIR = $DataDir
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = 'true'
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = (Resolve-Path $DocRoot).Path
$env:LABEL_STUDIO_ENABLE_LEGACY_API_TOKEN = 'true'   # token cố định trong credentials.json cho xla.detect.labelstudio
$env:LABEL_STUDIO_USERNAME = $cred.username           # chỉ dùng lần khởi tạo DB đầu tiên
$env:LABEL_STUDIO_PASSWORD = $cred.password
$env:LABEL_STUDIO_USER_TOKEN = $cred.token
$env:LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK = 'true'
$env:COLLECT_ANALYTICS = 'false'
$env:LABEL_STUDIO_COLLECT_ANALYTICS = 'false'
$env:SENTRY_DSN = ''
$env:FRONTEND_SENTRY_DSN = ''

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$p = Start-Process -FilePath $Exe -WindowStyle Hidden -PassThru `
    -ArgumentList @('start', '--internal-host', '127.0.0.1', '--port', "$Port", '--host', "http://localhost:$Port",
        '--data-dir', $DataDir, '--no-browser') `
    -RedirectStandardOutput (Join-Path $DataDir "logs\ls-$stamp.out.log") `
    -RedirectStandardError (Join-Path $DataDir "logs\ls-$stamp.err.log")
Write-Output "label-studio pid $($p.Id), http://localhost:$Port , log $DataDir\logs\ls-$stamp.*.log"
