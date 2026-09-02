param(
    [switch]$ResetSetup,
    [switch]$AdvancedSetup,
    [switch]$SetupOnly
)

$ErrorActionPreference = "Stop"

Write-Host "=== Z-Image Turbo Windows ==="
Write-Host ""

$root = $PSScriptRoot
$configDir = Join-Path $root "config"
$registryPath = Join-Path $configDir "model_registry.json"
$setupConfigPath = Join-Path $configDir "setup_config.json"
$sdBin = Join-Path $root "sd_bin"
$modelsDir = Join-Path $root "models"
$zimageDir = Join-Path $modelsDir "zimage"
$vaeDir = Join-Path $modelsDir "vae"
$llmDir = Join-Path $modelsDir "llm"
$loraDir = Join-Path $modelsDir "loras"
$downloadsDir = Join-Path $root "downloads"
$backendDownloadsDir = Join-Path $downloadsDir "backend"

function Ensure-Dir {
    param([string]$Path)
    if (!(Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Ensure-ProjectFolders {
    Ensure-Dir $configDir
    Ensure-Dir $sdBin
    Ensure-Dir $modelsDir
    Ensure-Dir $zimageDir
    Ensure-Dir $vaeDir
    Ensure-Dir $llmDir
    Ensure-Dir $loraDir
    Ensure-Dir $downloadsDir
    Ensure-Dir $backendDownloadsDir
}

function Load-Registry {
    if (!(Test-Path $registryPath)) {
        throw "Missing model registry: $registryPath"
    }
    return Get-Content -Raw -Path $registryPath | ConvertFrom-Json
}

function Save-SetupConfig {
    param([object]$Config)
    $Config | ConvertTo-Json -Depth 8 | Out-File -Encoding utf8 $setupConfigPath
}

function Load-SetupConfig {
    if (!(Test-Path $setupConfigPath)) {
        return $null
    }
    try {
        return Get-Content -Raw -Path $setupConfigPath | ConvertFrom-Json
    } catch {
        return $null
    }
}

function Download-FileWithProgress {
    param(
        [Parameter(Mandatory=$true)][string]$Url,
        [Parameter(Mandatory=$true)][string]$Destination,
        [Parameter(Mandatory=$true)][string]$Label
    )

    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    } catch {
    }

    $dir = Split-Path -Parent $Destination
    Ensure-Dir $dir

    try {
        $curlCmd = Get-Command curl.exe -ErrorAction SilentlyContinue
        if ($curlCmd -and $curlCmd.Source) {
            Write-Host ("{0}..." -f $Label)
            $args = @(
                "--location",
                "--fail",
                "--retry", "10",
                "--retry-delay", "5",
                "--retry-all-errors",
                "--connect-timeout", "30",
                "--speed-time", "30",
                "--speed-limit", "10240",
                "--progress-bar",
                "--header", "User-Agent: Mozilla/5.0",
                "--header", "Accept: */*"
            )
            if (Test-Path $Destination) {
                $args += @("--continue-at", "-")
            }
            $args += @("--output", $Destination, $Url)
            & $curlCmd.Source @args
            if ($LASTEXITCODE -ne 0) {
                throw "curl failed with exit code $LASTEXITCODE"
            }
            return
        }
    } catch {
        Write-Host "curl download failed, using fallback downloader."
        Write-Host ("Reason: {0}" -f $_.Exception.Message)
    }

    $headers = @{
        "User-Agent" = "Mozilla/5.0"
        "Accept" = "*/*"
    }
    Invoke-WebRequest -Uri $Url -OutFile $Destination -Headers $headers -MaximumRedirection 10 -ProxyUseDefaultCredentials
}

function Verify-Sha256 {
    param(
        [string]$Path,
        [string]$ExpectedSha256
    )
    if ([string]::IsNullOrWhiteSpace($ExpectedSha256)) {
        return $true
    }
    $actual = (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    $expected = $ExpectedSha256.Trim().ToLowerInvariant()
    if ($actual -ne $expected) {
        throw "SHA256 mismatch for $Path. Expected $expected but got $actual"
    }
    return $true
}

function Get-NvidiaInfo {
    $nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if (!$nvidiaSmi) {
        return $null
    }
    try {
        $line = & $nvidiaSmi.Source --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits 2>$null | Select-Object -First 1
        if (!$line) {
            return $null
        }
        $parts = $line -split ","
        return [pscustomobject]@{
            Vendor = "NVIDIA"
            Name = $parts[0].Trim()
            VramMb = [int]($parts[1].Trim())
            ComputeCapability = $parts[2].Trim()
            CudaAvailable = $true
        }
    } catch {
        return $null
    }
}

function Get-DisplayAdapters {
    try {
        return Get-CimInstance Win32_VideoController | ForEach-Object {
            [pscustomobject]@{
                Name = $_.Name
                AdapterRAM = $_.AdapterRAM
            }
        }
    } catch {
        return @()
    }
}

function Detect-Hardware {
    $nvidia = Get-NvidiaInfo
    if ($nvidia) {
        return [pscustomobject]@{
            Vendor = "NVIDIA"
            Name = $nvidia.Name
            VramMb = $nvidia.VramMb
            ComputeCapability = $nvidia.ComputeCapability
            CudaAvailable = $true
            BackendKind = "cuda12"
        }
    }

    $adapters = @(Get-DisplayAdapters)
    $adapterText = ($adapters | ForEach-Object { $_.Name }) -join "; "
    $firstAdapter = $adapters | Select-Object -First 1
    $adapterVramMb = 0
    if ($firstAdapter -and $firstAdapter.AdapterRAM) {
        try {
            $adapterVramMb = [int]($firstAdapter.AdapterRAM / 1MB)
        } catch {
            $adapterVramMb = 0
        }
    }

    if ($adapterText -match "AMD|Radeon") {
        return [pscustomobject]@{
            Vendor = "AMD"
            Name = $adapterText
            VramMb = $adapterVramMb
            ComputeCapability = ""
            CudaAvailable = $false
            BackendKind = "vulkan"
        }
    }
    if ($adapterText -match "Intel|Arc") {
        return [pscustomobject]@{
            Vendor = "Intel"
            Name = $adapterText
            VramMb = $adapterVramMb
            ComputeCapability = ""
            CudaAvailable = $false
            BackendKind = "vulkan"
        }
    }
    return [pscustomobject]@{
        Vendor = "CPU"
        Name = "No supported dedicated GPU detected"
        VramMb = 0
        ComputeCapability = ""
        CudaAvailable = $false
        BackendKind = "cpu"
    }
}


function Recommend-Profile {
    param([object]$Hardware)
    if ($Hardware.Vendor -eq "NVIDIA") {
        if ($Hardware.VramMb -lt 6144) {
            return "ultra_low_vram"
        }
        if ($Hardware.VramMb -lt 10240) {
            return "balanced"
        }
        return "high_end"
    }
    return "cpu_only"
}

function Get-ModelByProfile {
    param(
        [object]$Registry,
        [string]$ProfileId
    )
    $profile = $Registry.profiles.$ProfileId
    $modelId = $profile.default_model
    return [pscustomobject]@{
        Id = $modelId
        Model = $Registry.models.$modelId
    }
}

function Find-BackendExe {
    $sdCliExe = Join-Path $sdBin "sd-cli.exe"
    $sdOldExe = Join-Path $sdBin "sd.exe"
    if (Test-Path $sdCliExe) {
        return $sdCliExe
    }
    if (Test-Path $sdOldExe) {
        return $sdOldExe
    }
    return $null
}

function Get-LatestRelease {
    $headers = @{ "User-Agent" = "ZImage-Windows-Setup" }
    return Invoke-RestMethod -Uri "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest" -Headers $headers
}

function Get-AssetShaFromReleaseBody {
    param(
        [string]$Body,
        [string]$AssetName
    )
    if ([string]::IsNullOrWhiteSpace($Body)) {
        return ""
    }
    $escaped = [regex]::Escape($AssetName)
    $pattern = "(?is)$escaped.*?sha256:\s*([a-f0-9]{64})"
    $match = [regex]::Match($Body, $pattern)
    if ($match.Success) {
        return $match.Groups[1].Value
    }
    return ""
}

function Download-And-Extract-Asset {
    param(
        [object]$Release,
        [string]$NamePattern,
        [string]$Label
    )
    $asset = $Release.assets | Where-Object { $_.name -like $NamePattern } | Select-Object -First 1
    if (!$asset) {
        throw "Could not find release asset matching: $NamePattern"
    }

    $zipPath = Join-Path $backendDownloadsDir $asset.name
    Download-FileWithProgress -Url $asset.browser_download_url -Destination $zipPath -Label $Label

    $expectedSha = Get-AssetShaFromReleaseBody -Body $Release.body -AssetName $asset.name
    if ($expectedSha) {
        Verify-Sha256 -Path $zipPath -ExpectedSha256 $expectedSha | Out-Null
        Write-Host "Verified SHA256: $($asset.name)"
    } else {
        Write-Host "SHA256 not found in release notes for $($asset.name); continuing after successful download."
    }

    $extractDir = Join-Path $backendDownloadsDir ([IO.Path]::GetFileNameWithoutExtension($asset.name))
    if (Test-Path $extractDir) {
        Remove-Item -Recurse -Force $extractDir
    }
    Expand-Archive -LiteralPath $zipPath -DestinationPath $extractDir -Force
    return $extractDir
}

function Copy-BackendFiles {
    param([string]$ExtractDir)
    $files = Get-ChildItem -Path $ExtractDir -Recurse -File | Where-Object {
        $_.Name -in @("sd-cli.exe", "sd-server.exe", "stable-diffusion.dll") -or $_.Extension -eq ".dll"
    }
    foreach ($file in $files) {
        Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $sdBin $file.Name) -Force
    }
}

function Install-BackendAutomatically {
    param([string]$BackendKind)
    Write-Host ""
    Write-Host "Installing stable-diffusion.cpp backend automatically ($BackendKind)..."
    $release = Get-LatestRelease
    if ($BackendKind -eq "cuda12") {
        $cudaDir = Download-And-Extract-Asset -Release $release -NamePattern "sd-*-bin-win-cuda12-x64.zip" -Label "Downloading CUDA 12 backend"
        Copy-BackendFiles -ExtractDir $cudaDir
        $runtimeDir = Download-And-Extract-Asset -Release $release -NamePattern "cudart-sd-bin-win-cu12-x64.zip" -Label "Downloading CUDA runtime DLLs"
        Copy-BackendFiles -ExtractDir $runtimeDir
    } elseif ($BackendKind -eq "vulkan") {
        try {
            $vulkanDir = Download-And-Extract-Asset -Release $release -NamePattern "sd-*-bin-win-vulkan-x64.zip" -Label "Downloading Vulkan GPU backend"
            Copy-BackendFiles -ExtractDir $vulkanDir
        } catch {
            Write-Host "Vulkan package not found in release, falling back to AVX2 CPU backend..."
            $cpuDir = Download-And-Extract-Asset -Release $release -NamePattern "sd-*-bin-win-avx2-x64.zip" -Label "Downloading CPU (AVX2) backend"
            Copy-BackendFiles -ExtractDir $cpuDir
        }
    } else {
        $cpuDir = Download-And-Extract-Asset -Release $release -NamePattern "sd-*-bin-win-avx2-x64.zip" -Label "Downloading CPU (AVX2) backend"
        Copy-BackendFiles -ExtractDir $cpuDir
    }
}

function Ensure-PythonEnvironment {
    $venv = Join-Path $root "venv"
    if (!(Test-Path $venv)) {
        Write-Host "Creating Python virtual environment..."
        python -m venv venv
    } else {
        Write-Host "Virtual environment already exists."
    }

    $venvPython = Join-Path $venv "Scripts\python.exe"
    if (!(Test-Path $venvPython)) {
        throw "venv python not found at $venvPython"
    }

    Write-Host "Installing/updating Python requirements..."
    & $venvPython -m pip install --upgrade pip | Out-Host
    & $venvPython -m pip install gradio requests tqdm pillow | Out-Host
    return $venvPython
}

function Ensure-RegistryAsset {
    param(
        [object]$Asset,
        [string]$DestinationDir,
        [string]$Label
    )
    $destination = Join-Path $DestinationDir $Asset.filename
    if (Test-Path $destination) {
        Write-Host "$Label already exists: $destination"
        return $destination
    }
    Download-FileWithProgress -Url $Asset.url -Destination $destination -Label ("Downloading {0}" -f $Label)
    Verify-Sha256 -Path $destination -ExpectedSha256 $Asset.sha256 | Out-Null
    Write-Host "Downloaded $Label to: $destination"
    return $destination
}

function Test-SetupComplete {
    param(
        [object]$Config,
        [object]$Registry
    )
    if (!$Config) {
        return $false
    }
    if (!(Find-BackendExe)) {
        return $false
    }
    $modelId = $Config.model_id
    if (!$modelId) {
        return $false
    }
    $model = $Registry.models.$modelId
    if (!$model) {
        return $false
    }
    $required = @(
        (Join-Path $zimageDir $model.filename),
        (Join-Path $vaeDir $Registry.models."z-image-vae".filename),
        (Join-Path $llmDir $Registry.models."qwen-text-encoder-q4".filename)
    )
    foreach ($path in $required) {
        if (!(Test-Path $path)) {
            return $false
        }
    }
    return $true
}

function Run-SetupWizard {
    param([object]$Registry)

    Write-Host ""
    Write-Host "First-time setup wizard"
    Write-Host "This runs only when setup is incomplete or reset."
    Write-Host ""

    $hardware = Detect-Hardware
    $recommendedProfile = Recommend-Profile -Hardware $hardware
    $profile = $Registry.profiles.$recommendedProfile
    $modelInfo = Get-ModelByProfile -Registry $Registry -ProfileId $recommendedProfile

    Write-Host "Detected hardware:"
    Write-Host (" - Vendor : {0}" -f $hardware.Vendor)
    Write-Host (" - Device : {0}" -f $hardware.Name)
    if ($hardware.VramMb -gt 0) {
        Write-Host (" - VRAM   : {0:N1} GB" -f ($hardware.VramMb / 1024))
    }
    if ($hardware.ComputeCapability) {
        Write-Host (" - Compute: {0}" -f $hardware.ComputeCapability)
    }
    Write-Host ""
    Write-Host ("Recommended profile: {0} ({1})" -f $profile.display_name, $profile.target)
    Write-Host ("Recommended model  : {0}" -f $modelInfo.Model.display_name)
    Write-Host ""

    $selectedProfile = $recommendedProfile
    if ($AdvancedSetup) {
        Write-Host "Advanced setup: choose a profile or press Enter to accept the recommendation."
        Write-Host " 1) Ultra Low VRAM - 4GB GPUs"
        Write-Host " 2) Balanced - 6-8GB GPUs"
        Write-Host " 3) High-End - 10GB+ GPUs"
        Write-Host " 4) CPU Only"
        $profileChoice = Read-Host "Profile"
        switch ($profileChoice) {
            "1" { $selectedProfile = "ultra_low_vram" }
            "2" { $selectedProfile = "balanced" }
            "3" { $selectedProfile = "high_end" }
            "4" { $selectedProfile = "cpu_only" }
            default { $selectedProfile = $recommendedProfile }
        }
    } else {
        Write-Host "Using the recommended profile automatically."
    }

    $modelInfo = Get-ModelByProfile -Registry $Registry -ProfileId $selectedProfile
    $backendMode = "auto"
    if ($AdvancedSetup) {
        Write-Host ""
        Write-Host "Backend installation:"
        Write-Host " 1) Automatic download/install"
        Write-Host " 2) Manual binaries in sd_bin"
        $backendChoice = Read-Host "Choose 1 or 2"
        if ($backendChoice -eq "2") {
            $backendMode = "manual"
        }
    }

    $backendExe = Find-BackendExe
    if (!$backendExe -and $backendMode -eq "auto") {
        Install-BackendAutomatically -BackendKind $hardware.BackendKind
        $backendExe = Find-BackendExe
    }
    if (!$backendExe) {
        Write-Host ""
        Write-Host "No backend executable found."
        Write-Host "Place sd-cli.exe or sd.exe in: $sdBin"
        Write-Host "Then rerun start_zimage.bat."
        exit 1
    }

    $venvPython = Ensure-PythonEnvironment
    $modelPath = Ensure-RegistryAsset -Asset $modelInfo.Model -DestinationDir $zimageDir -Label $modelInfo.Model.display_name
    $vaePath = Ensure-RegistryAsset -Asset $Registry.models."z-image-vae" -DestinationDir $vaeDir -Label "Z-Image Turbo VAE"
    $llmPath = Ensure-RegistryAsset -Asset $Registry.models."qwen-text-encoder-q4" -DestinationDir $llmDir -Label "Qwen text encoder"

    $selectedModelPath = Join-Path $zimageDir "selected_model.txt"
    $modelInfo.Model.filename | Out-File -Encoding ascii $selectedModelPath

    $config = [ordered]@{
        schema_version = 1
        setup_complete = $true
        created_at = (Get-Date).ToString("o")
        updated_at = (Get-Date).ToString("o")
        profile_id = $selectedProfile
        model_id = $modelInfo.Id
        backend_mode = $backendMode
        backend_kind = $hardware.BackendKind
        backend_path = $backendExe
        venv_python = $venvPython
        paths = [ordered]@{
            model = $modelPath
            vae = $vaePath
            llm = $llmPath
        }
        hardware = [ordered]@{
            vendor = $hardware.Vendor
            name = $hardware.Name
            vram_mb = $hardware.VramMb
            compute_capability = $hardware.ComputeCapability
            cuda_available = $hardware.CudaAvailable
            cpu_threads = [Environment]::ProcessorCount
        }
        preferences = [ordered]@{
            launch_url = "http://127.0.0.1:9000"
        }
    }
    Save-SetupConfig -Config $config
    Write-Host ""
    Write-Host "Setup complete. Future launches will start immediately."
    return [pscustomobject]$config
}

function Wait-ForUiReady {
    param(
        [string]$Url,
        [System.Diagnostics.Process]$Process,
        [int]$TimeoutSeconds = 180
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if ($Process.HasExited) {
            throw "The UI closed before it was ready. Please check the messages above for the error."
        }
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                return
            }
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    throw "The UI did not become ready within $TimeoutSeconds seconds. Try running start_zimage.bat again."
}

function Launch-App {
    param(
        [object]$Config,
        [object]$Registry
    )
    $model = $Registry.models.($Config.model_id)
    if (!$model) {
        throw "Configured model not found in registry: $($Config.model_id)"
    }
    $selectedModelPath = Join-Path $zimageDir "selected_model.txt"
    $model.filename | Out-File -Encoding ascii $selectedModelPath
    $env:ZIMAGE_MODEL_NAME = $model.filename
    $env:ZIMAGE_PROFILE_ID = $Config.profile_id

    $venvPython = $Config.venv_python
    if (!$venvPython -or !(Test-Path $venvPython)) {
        $venvPython = Ensure-PythonEnvironment
        $Config.venv_python = $venvPython
        $Config.updated_at = (Get-Date).ToString("o")
        Save-SetupConfig -Config $Config
    }

    $uiScript = Join-Path $root "run_gradio_ui.py"
    if (!(Test-Path $uiScript)) {
        throw "run_gradio_ui.py not found."
    }
    Write-Host ""
    Write-Host "Launching Z-Image Turbo UI..."
    Write-Host ("Profile: {0}" -f $Config.profile_id)
    Write-Host ("Model  : {0}" -f $model.filename)
    Write-Host "Starting local server. The ready link will appear in a moment..."

    $launchUrl = "http://127.0.0.1:9000"
    $env:ZIMAGE_QUIET_LAUNCH = "1"
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $venvPython
    $startInfo.Arguments = "`"$uiScript`""
    $startInfo.WorkingDirectory = $root
    $startInfo.UseShellExecute = $false
    $process = [System.Diagnostics.Process]::Start($startInfo)

    Wait-ForUiReady -Url $launchUrl -Process $process
    Write-Host ""
    Write-Host "Ready. Open this link:"
    Write-Host ("URL    : {0}" -f $launchUrl)
    $process.WaitForExit()
}

Ensure-ProjectFolders
$registry = Load-Registry

if ($ResetSetup -and (Test-Path $setupConfigPath)) {
    Remove-Item -LiteralPath $setupConfigPath -Force
}

$config = Load-SetupConfig
if (!(Test-SetupComplete -Config $config -Registry $registry)) {
    $config = Run-SetupWizard -Registry $registry
} else {
    Write-Host "Setup already complete. Starting app."
}

if ($SetupOnly) {
    Write-Host "Setup verification complete. Skipping app launch because -SetupOnly was used."
} else {
    Launch-App -Config $config -Registry $registry
}
