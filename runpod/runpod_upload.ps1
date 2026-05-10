# runpod_upload.ps1 — Upload project ke RunPod
# ─────────────────────────────────────────────
# Cara pakai:
#   .\runpod_upload.ps1              # upload source + data/labeled
#   .\runpod_upload.ps1 -SkipData   # upload source saja (data sudah ada di server)
#   .\runpod_upload.ps1 -DataOnly   # upload data/labeled saja (source sudah ada)
#
# Pastikan runpod.env sudah diisi sebelum menjalankan ini.

param(
    [switch]$SkipData,
    [switch]$DataOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Path ──────────────────────────────────────────────────────────────────────
$RUNPOD_DIR = $PSScriptRoot                        # .../Riset_pemodelan/runpod/
$LOCAL      = Split-Path $PSScriptRoot -Parent     # .../Riset_pemodelan/

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
    Write-Error "Belum mengisi runpod.env! Edit RUNPOD_HOST dan RUNPOD_PORT terlebih dahulu."
    exit 1
}
foreach ($k in @("RUNPOD_HOST","RUNPOD_PORT","RUNPOD_USER","REMOTE_DIR")) {
    if (-not $config[$k]) { Write-Error "runpod.env: $k belum diisi."; exit 1 }
}

# ── Bangun argumen SSH / SCP ──────────────────────────────────────────────────
$sshExtra = @("-p", $RPORT, "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=no")
$scpExtra = @("-P", $RPORT, "-o", "StrictHostKeyChecking=no")
if ($RKEY) { $sshExtra += @("-i", $RKEY); $scpExtra += @("-i", $RKEY) }
$target = "${RUSER}@${RHOST}"

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  RunPod Upload — Riset Pemodelan" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Target : $target (port $RPORT)"
Write-Host "  Remote : $REMOTE"
Write-Host "  Local  : $LOCAL"
Write-Host ""

# ── Helper: SSH ───────────────────────────────────────────────────────────────
function Invoke-SSH([string]$cmd) {
    & ssh @sshExtra $target $cmd
    if ($LASTEXITCODE -ne 0) { throw "SSH gagal: $cmd" }
}

# ── Helper: pack lalu scp ─────────────────────────────────────────────────────
function Upload-Archive {
    param(
        [string]$Label,
        [string]$TarRoot,
        [string]$TarTarget,
        [string]$RemoteDest,
        [string[]]$Excludes = @()
    )
    Write-Host "  Packing $Label..." -ForegroundColor Yellow
    $tmp = [System.IO.Path]::GetTempFileName() + ".tar.gz"
    try {
        $tarArgs = @("-czf", $tmp, "-C", $TarRoot)
        foreach ($ex in $Excludes) { $tarArgs += "--exclude=$ex" }
        $tarArgs += $TarTarget

        & tar @tarArgs
        if ($LASTEXITCODE -ne 0) { throw "tar gagal untuk $Label" }

        $sizeMB = [math]::Round((Get-Item $tmp).Length / 1MB, 1)
        Write-Host "  Uploading $Label ($sizeMB MB)..." -ForegroundColor Yellow

        $tmpRemote = "/tmp/runpod_upload_$(Get-Date -Format 'HHmmss').tar.gz"
        & scp @scpExtra $tmp "${target}:${tmpRemote}"
        if ($LASTEXITCODE -ne 0) { throw "SCP gagal untuk $Label" }

        Invoke-SSH "mkdir -p $RemoteDest && tar -xzf $tmpRemote -C $RemoteDest && rm -f $tmpRemote"
        Write-Host "  $Label OK" -ForegroundColor Green
    } finally {
        Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    }
}

# ── Step 1: Buat direktori remote ─────────────────────────────────────────────
Write-Host "[1/3] Buat direktori di server..." -ForegroundColor Green
Invoke-SSH "mkdir -p $REMOTE/data/labeled $REMOTE/data/processed $REMOTE/models $REMOTE/logs"

# ── Step 2: Upload source code ────────────────────────────────────────────────
if (-not $DataOnly) {
    Write-Host "[2/3] Upload source code..." -ForegroundColor Green
    Upload-Archive `
        -Label      "source code" `
        -TarRoot    $LOCAL `
        -TarTarget  "." `
        -RemoteDest $REMOTE `
        -Excludes   @(".git", "__pycache__", "*.pyc", "*.log", "data", "models", ".claude", "*.bak")
} else {
    Write-Host "[2/3] Skip source code (--DataOnly)" -ForegroundColor Gray
}

# ── Step 3: Upload data/labeled ───────────────────────────────────────────────
if (-not $SkipData) {
    $labeledPath = Join-Path $LOCAL "data\labeled"
    if (Test-Path $labeledPath) {
        $fileCount = (Get-ChildItem $labeledPath -File).Count
        Write-Host "[3/3] Upload data/labeled ($fileCount file, mungkin butuh beberapa menit)..." -ForegroundColor Green
        Upload-Archive `
            -Label      "data/labeled" `
            -TarRoot    (Join-Path $LOCAL "data") `
            -TarTarget  "labeled" `
            -RemoteDest "$REMOTE/data"
    } else {
        Write-Warning "[3/3] data\labeled tidak ditemukan di lokal — skip."
        Write-Host "       Pastikan pipeline 01-04 sudah dijalankan terlebih dahulu." -ForegroundColor Yellow
    }
} else {
    Write-Host "[3/3] Skip data/labeled (--SkipData)" -ForegroundColor Gray
}

# ── Selesai ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Upload selesai!" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Langkah selanjutnya:"
Write-Host "  .\runpod_vscode.ps1          # buka VS Code Remote" -ForegroundColor White
Write-Host "  .\runpod_vscode.ps1 -Setup   # setup dulu lalu buka VS Code" -ForegroundColor White
Write-Host ""
