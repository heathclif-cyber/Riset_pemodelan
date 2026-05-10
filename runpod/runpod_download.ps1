# runpod_download.ps1 — Download hasil training dari RunPod ke lokal
# ──────────────────────────────────────────────────────────────────
# Cara pakai:
#   .\runpod_download.ps1              # download models/ + logs/
#   .\runpod_download.ps1 -ModelsOnly  # hanya models/
#   .\runpod_download.ps1 -LogsOnly    # hanya logs/ (cek progress tanpa download model)

param(
    [switch]$ModelsOnly,
    [switch]$LogsOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Path ──────────────────────────────────────────────────────────────────────
$RUNPOD_DIR = $PSScriptRoot
$LOCAL      = Split-Path $PSScriptRoot -Parent

# ── Baca runpod.env ───────────────────────────────────────────────────────────
$envFile = "$RUNPOD_DIR\runpod.env"
if (-not (Test-Path $envFile)) {
    Write-Error "File runpod.env tidak ditemukan di $RUNPOD_DIR"
    exit 1
}

$config = @{}
Get-Content $envFile | Where-Object { $_ -match "^\s*[A-Z]" } | ForEach-Object {
    $parts = $_ -split "=", 2
    if ($parts.Length -eq 2) { $config[$parts[0].Trim()] = $parts[1].Trim() }
}

$RHOST  = $config["RUNPOD_HOST"]
$RPORT  = $config["RUNPOD_PORT"]
$RUSER  = $config["RUNPOD_USER"]
$RKEY   = $config["RUNPOD_SSH_KEY"]
$REMOTE = $config["REMOTE_DIR"]

if ($RHOST -like "GANTI*" -or $RPORT -like "GANTI*") {
    Write-Error "Belum mengisi runpod.env!"
    exit 1
}

$sshExtra = @("-p", $RPORT, "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=no")
$scpExtra = @("-P", $RPORT, "-o", "StrictHostKeyChecking=no")
if ($RKEY) { $sshExtra += @("-i", $RKEY); $scpExtra += @("-i", $RKEY) }
$target = "${RUSER}@${RHOST}"

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  RunPod Download — Riset Pemodelan" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Source : $target (port $RPORT)"
Write-Host "  Remote : $REMOTE"
Write-Host "  Local  : $LOCAL"
Write-Host ""

# ── Helper: cek status training ───────────────────────────────────────────────
function Get-TrainingStatus {
    $result = & ssh @sshExtra $target "tmux has-session -t lstm_train 2>/dev/null && echo RUNNING || echo DONE"
    return $result.Trim()
}

# ── Helper: download folder dari server ──────────────────────────────────────
function Download-Folder {
    param([string]$FolderName, [string]$RemoteParent, [string]$LocalDest)

    Write-Host "  Memeriksa $FolderName di server..." -ForegroundColor Yellow
    $exists = & ssh @sshExtra $target "[ -d $RemoteParent/$FolderName ] && echo yes || echo no"
    if ($exists.Trim() -ne "yes") {
        Write-Warning "  $FolderName tidak ditemukan di server — skip."
        return
    }

    $sizeInfo = & ssh @sshExtra $target "du -sh $RemoteParent/$FolderName 2>/dev/null | cut -f1"
    Write-Host "  Packing $FolderName ($($sizeInfo.Trim())) di server..." -ForegroundColor Yellow

    $tmpRemote = "/tmp/dl_${FolderName}_$(Get-Date -Format 'HHmmss').tar.gz"
    & ssh @sshExtra $target "tar -czf $tmpRemote -C $RemoteParent $FolderName"
    if ($LASTEXITCODE -ne 0) { throw "Gagal pack $FolderName di server" }

    $tmp = [System.IO.Path]::GetTempFileName() + ".tar.gz"
    try {
        Write-Host "  Downloading $FolderName..." -ForegroundColor Yellow
        & scp @scpExtra "${target}:${tmpRemote}" $tmp
        if ($LASTEXITCODE -ne 0) { throw "SCP download gagal untuk $FolderName" }

        $sizeMB = [math]::Round((Get-Item $tmp).Length / 1MB, 1)

        # Backup folder lokal yang sudah ada
        $localFolder = Join-Path $LocalDest $FolderName
        if (Test-Path $localFolder) {
            $backupName = "${FolderName}_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
            Write-Host "  Backup lokal → $backupName" -ForegroundColor Gray
            Rename-Item $localFolder (Join-Path $LocalDest $backupName)
        }

        New-Item -ItemType Directory -Force -Path $LocalDest | Out-Null
        & tar -xzf $tmp -C $LocalDest
        if ($LASTEXITCODE -ne 0) { throw "Ekstrak tar gagal untuk $FolderName" }

        & ssh @sshExtra $target "rm -f $tmpRemote" | Out-Null
        Write-Host "  $FolderName OK ($sizeMB MB → $LocalDest\$FolderName)" -ForegroundColor Green
    } finally {
        Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    }
}

# ── Cek status training ───────────────────────────────────────────────────────
Write-Host "Mengecek status training di server..." -ForegroundColor Gray
try {
    $status = Get-TrainingStatus
    if ($status -eq "RUNNING") {
        Write-Host "  WARNING: Training masih berjalan (tmux: lstm_train)!" -ForegroundColor Yellow
        Write-Host "  Model yang didownload mungkin belum final."
        Write-Host ""
        $confirm = Read-Host "  Lanjutkan download? (y/N)"
        if ($confirm -notmatch "^[yY]") {
            Write-Host "  Dibatalkan. Tunggu training selesai lalu jalankan ulang."
            exit 0
        }
    } else {
        Write-Host "  Training sudah selesai." -ForegroundColor Green
    }
} catch {
    Write-Host "  Tidak bisa cek status (lanjutkan download)." -ForegroundColor Gray
}

# ── Download ──────────────────────────────────────────────────────────────────
Write-Host ""
if ($LogsOnly) {
    Download-Folder -FolderName "logs"   -RemoteParent $REMOTE -LocalDest $LOCAL
} elseif ($ModelsOnly) {
    Download-Folder -FolderName "models" -RemoteParent $REMOTE -LocalDest $LOCAL
} else {
    Download-Folder -FolderName "models" -RemoteParent $REMOTE -LocalDest $LOCAL
    Download-Folder -FolderName "logs"   -RemoteParent $REMOTE -LocalDest $LOCAL
}

# ── Ringkasan ─────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Download selesai!" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

$modelsDir = Join-Path $LOCAL "models"
if (Test-Path $modelsDir) {
    Write-Host ""
    Write-Host "  File model yang didownload:"
    Get-ChildItem $modelsDir -Recurse -Include "*.pt","*.pkl","*.json" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 10 |
        ForEach-Object { Write-Host "    .$($_.FullName.Replace($LOCAL,''))" -ForegroundColor White }
}
Write-Host ""
