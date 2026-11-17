# Biến môi trường XLA: mọi cache, venv, model tải về nằm ở ổ D (không để ổ C).
# Dùng: . D:\XLA\scripts\env.ps1   (dot-source, trong PowerShell)
# Rồi: uv run --no-sync python ...   (lane khác lane serial KHÔNG để uv tự sync venv chung)

$env:UV_CACHE_DIR = 'D:\DevCache\uv'
$env:UV_PROJECT_ENVIRONMENT = 'D:\DevCache\venvs\xla'
$env:UV_TOOL_DIR = 'D:\DevCache\uv-tools'
$env:UV_PYTHON_INSTALL_DIR = 'D:\DevCache\uv-python'
$env:PIP_CACHE_DIR = 'D:\DevCache\pip'
$env:TORCH_HOME = 'D:\DevCache\torch'
$env:HF_HOME = 'D:\DevCache\hf'
$env:YOLO_CONFIG_DIR = 'D:\DevCache\ultralytics'
$env:XLA_DATA_ROOT = 'D:\DevCache\datasets\xla'
$env:PYTHONUTF8 = '1'
# file tạm (pytest tmp_path, tempfile) cũng ở ổ D
$env:TMP = 'D:\DevCache\tmp'
$env:TEMP = 'D:\DevCache\tmp'

foreach ($d in @($env:UV_CACHE_DIR, $env:UV_TOOL_DIR, $env:PIP_CACHE_DIR, $env:TORCH_HOME, $env:HF_HOME,
        $env:YOLO_CONFIG_DIR, $env:XLA_DATA_ROOT, 'D:\DevCache\locks', $env:TMP)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
}
