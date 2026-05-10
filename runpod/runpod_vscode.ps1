# runpod_vscode.ps1 — Buka project di VS Code dengan GPU RunPod
# ──────────────────────────────────────────────────────────────
# Cara pakai:
#   .\runpod_vscode.ps1              # update SSH config + buka VS Code remote
#   .\runpod_vscode.ps1 -Setup       # setup server dulu, lalu buka VS Code
#   .\runpod_vscode.ps1 -Upload      # upload project + setup, lalu buka VS Code
#
# Prasyarat: extension "Remote - SSH" sudah terinstall di VS Code

param(
    [switch]$Setup,
    [switch]$Upload
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
    Write-Error "Belum mengisi runpod.env! Edit RUNPOD_HOST dan RUNPOD_PORT terlebih dahulu."
    exit 1
}

$sshArgs = @("-p", $RPORT, "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10")
if ($RKEY) { $sshArgs += @("-i", $RKEY) }
$target = "${RUSER}@${RHOST}"

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  RunPod VS Code Remote" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Pod    : $target (port $RPORT)"
Write-Host "  Remote : $REMOTE"
Write-Host ""

# ── 1. Cek koneksi ───────────────────────────────────────────────────────────
Write-Host "[1] Cek koneksi ke RunPod..." -ForegroundColor Green
try {
    $ping = & ssh @sshArgs $target "echo OK" 2>&1
    if ($ping -notmatch "OK") { throw "Tidak ada respons" }
    Write-Host "  Koneksi OK" -ForegroundColor Green
} catch {
    Write-Error "Gagal konek ke $target port $RPORT. Pastikan pod sudah berjalan dan info SSH benar."
    exit 1
}

# ── 2. Upload (opsional) ──────────────────────────────────────────────────────
if ($Upload) {
    Write-Host "[2] Upload project..." -ForegroundColor Green
    & "$RUNPOD_DIR\runpod_upload.ps1"
    if ($LASTEXITCODE -ne 0) { Write-Error "Upload gagal"; exit 1 }
} else {
    Write-Host "[2] Skip upload (gunakan -Upload jika perlu upload ulang)" -ForegroundColor Gray
}

# ── 3. Setup server (opsional) ────────────────────────────────────────────────
if ($Setup -or $Upload) {
    Write-Host "[3] Setup environment di server..." -ForegroundColor Green
    & ssh @sshArgs $target "bash $REMOTE/runpod/runpod_setup.sh"
    if ($LASTEXITCODE -ne 0) { Write-Error "Setup gagal"; exit 1 }
} else {
    Write-Host "[3] Skip setup (gunakan -Setup jika perlu install deps)" -ForegroundColor Gray
}

# ── 4. Update ~/.ssh/config ───────────────────────────────────────────────────
Write-Host "[4] Update SSH config untuk VS Code..." -ForegroundColor Green

$sshDir    = "$env:USERPROFILE\.ssh"
$sshConfig = "$sshDir\config"
if (-not (Test-Path $sshDir)) { New-Item -ItemType Directory -Path $sshDir -Force | Out-Null }

$keyLine = if ($RKEY) { "`n    IdentityFile $RKEY" } else { "" }
$newBlock = @"
Host runpod
    HostName $RHOST
    Port $RPORT
    User $RUSER
    StrictHostKeyChecking no
    ServerAliveInterval 60
    ServerAliveCountMax 10$keyLine
"@

if (Test-Path $sshConfig) {
    $existing = Get-Content $sshConfig -Raw
    $cleaned  = $existing -replace "(?ms)^Host runpod\r?\n(    [^\r\n]+\r?\n)*", ""
    Set-Content -Path $sshConfig -Value ($cleaned.TrimEnd() + "`n`n" + $newBlock) -Encoding UTF8
} else {
    Set-Content -Path $sshConfig -Value $newBlock -Encoding UTF8
}

Write-Host "  SSH config diupdate: $sshConfig" -ForegroundColor Green

# ── 5. Cek extension Remote - SSH ────────────────────────────────────────────
Write-Host "[5] Cek VS Code..." -ForegroundColor Green
$codeCmd = Get-Command "code" -ErrorAction SilentlyContinue
if (-not $codeCmd) {
    Write-Warning "  VS Code CLI 'code' tidak ditemukan di PATH."
    Write-Host ""
    Write-Host "  Buka VS Code manual:" -ForegroundColor Yellow
    Write-Host "    Ctrl+Shift+P → 'Remote-SSH: Connect to Host' → pilih 'runpod'" -ForegroundColor White
    exit 0
}

$extList = & code --list-extensions 2>$null
if ($extList -notcontains "ms-vscode-remote.remote-ssh") {
    Write-Host "  WARNING: Extension 'Remote - SSH' belum terinstall!" -ForegroundColor Yellow
    $install = Read-Host "  Install sekarang? (Y/n)"
    if ($install -notmatch "^[nN]") {
        & code --install-extension ms-vscode-remote.remote-ssh
        Write-Host "  Extension terinstall." -ForegroundColor Green
    }
}

# ── 6. Buka VS Code Remote ────────────────────────────────────────────────────
Write-Host "[6] Membuka VS Code Remote SSH → $REMOTE..." -ForegroundColor Green
& code --remote "ssh-remote+runpod" $REMOTE

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  VS Code terbuka dengan koneksi RunPod!" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Di terminal VS Code (Ctrl+`):"
Write-Host ""
Write-Host "  Training 5 coins:"
Write-Host "    python pipeline/06_train_lstm.py" -ForegroundColor White
Write-Host ""
Write-Host "  Training semua 20 coins:"
Write-Host "    python pipeline/06_train_lstm.py --all" -ForegroundColor White
Write-Host ""
Write-Host "  Cek GPU:"
Write-Host "    python -c ""import torch; print(torch.cuda.get_device_name(0))""" -ForegroundColor White
Write-Host ""
Write-Host "  TIP: Ctrl+Shift+P → 'Python: Select Interpreter' → pilih /usr/bin/python3"
Write-Host ""
