[CmdletBinding()]
param(
    [ValidateSet('subtitles', 'review', 'media')]
    [string]$Mode = 'media',
    [switch]$PreparedAudio,
    [string]$Language
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$configPath = Join-Path $root 'workflow-config.json'
$config = @{}
if (Test-Path -LiteralPath $configPath -PathType Leaf) {
    try {
        $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json -AsHashtable
        if ($config -isnot [System.Collections.IDictionary]) { throw 'Expected an object' }
    } catch {
        throw '本地 workflow-config.json 格式无效，请按样例修正。'
    }
}

function Get-ApplicationPath([string]$Name) {
    $command = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($command) { $command.Source }
}

function Test-File([string]$Path) {
    $Path -and [IO.Path]::IsPathFullyQualified($Path) -and
        (Test-Path -LiteralPath $Path -PathType Leaf)
}

function Test-Starts([string]$Path, [string]$Argument) {
    if (-not (Test-File $Path)) { return $false }
    try {
        & $Path $Argument 2>&1 | Out-Null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

$work = [string]$config.work_root
$validWork = $false
if ($work -and [IO.Path]::IsPathFullyQualified($work)) {
    $work = [IO.Path]::GetFullPath($work)
    if (Test-Path -LiteralPath $work -PathType Container) {
        $work = (Resolve-Path -LiteralPath $work).Path
    }
    $relative = [IO.Path]::GetRelativePath($root, $work)
    $outside = [IO.Path]::IsPathFullyQualified($relative) -or
        $relative -eq '..' -or $relative.StartsWith("..$([IO.Path]::DirectorySeparatorChar)")
    $validWork = $outside -and -not (Test-Path -LiteralPath $work -PathType Leaf)
}
$status = [ordered]@{ work_root = [bool]$validWork }
$found = [ordered]@{}

if ($Mode -ne 'subtitles') {
    $python = [string]$config.python_executable
    if (-not $python) {
        $venv = Join-Path $root '.venv/Scripts/python.exe'
        $launcher = Get-ApplicationPath 'py.exe'
        if (Test-File $venv) {
            $python = $venv
        } elseif ($launcher) {
            $python = & $launcher -3.12 -B -c 'import sys; print(sys.executable)' 2>$null |
                Select-Object -Last 1
        }
        if (-not $python) { $python = Get-ApplicationPath 'python.exe' }
    }
    $status.python = $false
    if (Test-File $python) {
        try {
            & $python -B -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,12) else 1)' 2>&1 | Out-Null
            $status.python = $LASTEXITCODE -eq 0
        } catch {}
    }
    if ($status.python) { $found.python_executable = $python }

    if ($Mode -eq 'media' -or -not $PreparedAudio) {
        $ffmpegBin = [string]$config.ffmpeg_bin
        $ffmpeg = $null
        $ffprobe = $null
        if (-not $ffmpegBin) {
            $bundled = Join-Path $root 'bin'
            $ffmpeg = Get-ApplicationPath 'ffmpeg.exe'
            $ffprobe = Get-ApplicationPath 'ffprobe.exe'
            if ($ffmpeg -and $ffprobe -and
                (Split-Path -Parent $ffmpeg) -eq (Split-Path -Parent $ffprobe)) {
                $ffmpegBin = Split-Path -Parent $ffmpeg
            } elseif ((Test-File (Join-Path $bundled 'ffmpeg.exe')) -and
                      (Test-File (Join-Path $bundled 'ffprobe.exe'))) {
                $ffmpegBin = $bundled
            }
        }
        if ($ffmpegBin) {
            $ffmpeg = Join-Path $ffmpegBin 'ffmpeg.exe'
            $ffprobe = Join-Path $ffmpegBin 'ffprobe.exe'
        }
        $status.ffmpeg = (Test-Starts $ffmpeg '-version') -and (Test-Starts $ffprobe '-version')
        if ($status.ffmpeg -and $ffmpegBin) { $found.ffmpeg_bin = $ffmpegBin }
    }

    $geminiPath = Join-Path $root 'gemini-config.json'
    $status.gemini_config = $false
    if (Test-Path -LiteralPath $geminiPath -PathType Leaf) {
        try {
            $gemini = Get-Content -LiteralPath $geminiPath -Raw | ConvertFrom-Json -AsHashtable
            $status.gemini_config = ($gemini -is [System.Collections.IDictionary]) -and
                (-not [string]::IsNullOrWhiteSpace($env:GEMINI_API_KEY) -or
                 -not [string]::IsNullOrWhiteSpace([string]$gemini.api_key))
        } catch {}
    }

    if ($Mode -eq 'media') {
        foreach ($module in @('faster_whisper', 'huggingface_hub', 'numpy', 'onnxruntime')) {
            $status[$module] = $false
            if ($status.python) {
                try {
                    & $python -B -c "import $module" 2>&1 | Out-Null
                    $status[$module] = $LASTEXITCODE -eq 0
                } catch {}
            }
        }
        $xxl = [string]$config.xxl_executable
        if (-not $xxl) { $xxl = Get-ApplicationPath 'faster-whisper-xxl.exe' }
        $status.xxl_executable = [bool](Test-File $xxl)
        if ($status.xxl_executable) { $found.xxl_executable = $xxl }
        $models = [string]$config.xxl_models
        if (-not $models -and $status.xxl_executable) {
            $models = Join-Path (Split-Path -Parent $xxl) '_models'
        }
        $preset = if ($Language -match '^(ja|ko|zh)(-|$)') { 'east-asian' } else { 'western' }
        $whisper = Get-Content -LiteralPath (Join-Path $root "config/$preset.json") -Raw |
            ConvertFrom-Json -AsHashtable
        $status.xxl_model = [bool]($models -and
            (Test-File (Join-Path $models "faster-whisper-$($whisper.model)/model.bin")))
        if ($status.xxl_model) { $found.xxl_models = $models }
        if ($Language) {
            $status.ctc_model = $false
            $registryPath = Join-Path $root 'alignment-models.json'
            if (Test-Path -LiteralPath $registryPath -PathType Leaf) {
                try {
                    $registry = Get-Content -LiteralPath $registryPath -Raw | ConvertFrom-Json -AsHashtable
                    $entry = $registry[$Language]
                    if ($entry.directory) {
                        $model = Join-Path $root $entry.directory
                        $status.ctc_model = [bool]((Test-File (Join-Path $model 'model_int8.onnx')) -and
                            (Test-File (Join-Path $model 'vocab.json')) -and
                            (Test-File (Join-Path $model 'config.json')))
                    }
                } catch {}
            }
        }
    }
}

[ordered]@{ mode = $Mode; checks = $status; discovered = $found } | ConvertTo-Json -Depth 4
