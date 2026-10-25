# Итоговый запуск решения. Пример:
#   .\run_final.ps1 -InputDir "C:\photos" -Output predictions.csv
# Без NVIDIA: -Device cpu
# Если на видеокарте возникнет ошибка CUDA: добавь -NoGraphs (чуть медленнее, без CUDA graphs).
param(
    [Parameter(Mandatory = $true)][string]$InputDir,
    [string]$Output = "predictions.csv",
    [string]$Device = "cuda",
    [string]$Benchmark = "",
    [switch]$NoGraphs
)
$ErrorActionPreference = "Stop"
# Относительные пути считаются от папки, откуда запущен скрипт, а не от папки проекта.
$InputDir = (Resolve-Path -LiteralPath $InputDir).Path
$Output = [System.IO.Path]::GetFullPath([System.IO.Path]::Combine((Get-Location).Path, $Output))
$extra = @()
if ($Benchmark -ne "") {
    $extra += @("--benchmark", [System.IO.Path]::GetFullPath([System.IO.Path]::Combine((Get-Location).Path, $Benchmark)))
}
if ($Device -like "cuda*" -and -not $NoGraphs) {
    $extra += @("--cuda-graphs")
}
Set-Location -LiteralPath $PSScriptRoot

python -m alpr.infer `
    --input $InputDir `
    --output $Output `
    --detector-backend yolo `
    --detector models/yolo11n_real_20261024/weights/best.pt `
    --white-detector models/yolo11n_white_20261024/weights/best.pt `
    --ocr models/ocr_synth2_20261025/best.pt `
    --char-threshold 0.4 `
    --type1-ocr models/ocr_type1_small_20261025/best.pt `
    --type1-char-threshold 0.5 `
    --multiscale-type1 `
    --threads 1 `
    --keep-nonvehicle `
    --direct-yolo `
    --gate-second-pass `
    --decode-max-side 2048 `
    --base-extra-size 960 `
    --base-extra-gate large_or_empty `
    --length-repair `
    --prefetch `
    --device $Device @extra
exit $LASTEXITCODE
