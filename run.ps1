# Launch the OpenCV desktop UI.
# Resolution order: $env:VISION_PYTHON override -> .venv\Scripts\python.exe -> python on PATH.
# Usage: .\run.ps1 --camera 0 --no-voice
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

$py = $env:VISION_PYTHON
if (-not $py) {
    $venv = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $venv) {
        $py = $venv
    } else {
        $py = 'python'
    }
}
if (-not $Rest) { $Rest = @() }
& $py (Join-Path $PSScriptRoot 'main.py') @Rest
exit $LASTEXITCODE
