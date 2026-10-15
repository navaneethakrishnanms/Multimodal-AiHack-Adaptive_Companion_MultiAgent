# Start Study Companion (Windows). Uses the conda env named in $env:SC_ENV (default torch_gpu), or the active python.
$envName = if ($env:SC_ENV) { $env:SC_ENV } else { "torch_gpu" }
$py = Join-Path $env:USERPROFILE "anaconda3\envs\$envName\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
& $py (Join-Path $PSScriptRoot "..\server.py")
