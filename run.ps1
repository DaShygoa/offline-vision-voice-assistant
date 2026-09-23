# Launch with vision_ai env (avoids base python missing openvino/onnxruntime)
# Usage:  .\run.ps1 --camera 2 --no-voice
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$py = 'C:\Users\admin\anaconda3\envs\vision_ai\python.exe'
if (-not (Test-Path -LiteralPath $py)) {
    Write-Error "vision_ai python not found: $py"
    exit 1
}
if (-not $Rest) { $Rest = @() }
& $py (Join-Path $PSScriptRoot 'main.py') @Rest
exit $LASTEXITCODE
