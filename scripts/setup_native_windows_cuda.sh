#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Native Windows CUDA environment bootstrapper
#
# Run from WSL/bash.
#
# - Invokes Windows PowerShell non-interactively.
# - Finds or installs Windows Python 3.12.
# - Creates .venv-win inside this repository.
# - Installs the same CUDA/PyTorch stack used by our QLoRA work.
# - Verifies that the process is genuinely native Windows.
#
# No interactive PowerShell or Windows Terminal required.
# ============================================================

REPO_WSL="${1:-$PWD}"

REPO_WSL="$(
    cd "$REPO_WSL"
    pwd -P
)"

if [[ ! -f "$REPO_WSL/requirements.txt" ]]; then
    echo "ERROR: requirements.txt not found:"
    echo "  $REPO_WSL"
    exit 2
fi

if ! command -v wslpath >/dev/null 2>&1; then
    echo "ERROR: wslpath not available."
    echo "Run this script from WSL."
    exit 2
fi

# ------------------------------------------------------------
# Windows PowerShell
# ------------------------------------------------------------

POWERSHELL="$(
    command -v powershell.exe 2>/dev/null || true
)"

if [[ -z "$POWERSHELL" ]]; then
    POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
fi

if [[ ! -x "$POWERSHELL" ]]; then
    echo "ERROR: powershell.exe not found."
    exit 2
fi

REPO_WIN="$(
    wslpath -w "$REPO_WSL"
)"

echo
echo "============================================"
echo " Native Windows CUDA Bootstrap"
echo "============================================"
echo
echo "WSL repo:"
echo "  $REPO_WSL"
echo
echo "Windows repo:"
echo "  $REPO_WIN"
echo
echo "PowerShell:"
echo "  $POWERSHELL"
echo

# ------------------------------------------------------------
# Ask PowerShell to find/install Windows Python 3.12.
#
# We emit a marker because winget may print additional output.
# ------------------------------------------------------------

echo "[1/5] Finding Windows Python 3.12..."

PY_WIN="$(
    "$POWERSHELL" \
        -NoLogo \
        -NoProfile \
        -NonInteractive \
        -ExecutionPolicy Bypass \
        -Command '
$ErrorActionPreference = "Stop"

function Find-Python312 {

    $launcher = Get-Command py.exe `
        -ErrorAction SilentlyContinue

    if ($launcher) {

        try {

            $resolved = & $launcher.Source `
                -3.12 `
                -c "import sys; print(sys.executable)" `
                2>$null

            if (
                $LASTEXITCODE -eq 0 `
                -and $resolved
            ) {
                return $resolved.ToString().Trim()
            }

        } catch {
        }
    }

    $candidate = Get-ChildItem `
        "$env:LOCALAPPDATA\Programs\Python\Python312*\python.exe" `
        -File `
        -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending |
        Select-Object -First 1

    if ($candidate) {
        return $candidate.FullName
    }

    $candidate = Get-ChildItem `
        "$env:ProgramFiles\Python312*\python.exe" `
        -File `
        -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending |
        Select-Object -First 1

    if ($candidate) {
        return $candidate.FullName
    }

    return $null
}

$Python = Find-Python312

if (-not $Python) {

    Write-Host "Python 3.12 not found."
    Write-Host "Installing per-user with winget..."

    $winget = Get-Command winget.exe `
        -ErrorAction SilentlyContinue

    if (-not $winget) {
        throw "winget.exe not found. Microsoft App Installer is required."
    }

    & $winget.Source install `
        --id Python.Python.3.12 `
        --exact `
        --scope user `
        --silent `
        --accept-package-agreements `
        --accept-source-agreements

    if ($LASTEXITCODE -ne 0) {
        throw "winget failed to install Python 3.12."
    }

    Start-Sleep -Seconds 2

    $Python = Find-Python312
}

if (-not $Python) {
    throw "Python 3.12 could not be located."
}

Write-Output ("__PYTHON__=" + $Python)
' |
        tr -d '\r' |
        sed -n 's/^__PYTHON__=//p' |
        tail -n 1
)"

if [[ -z "$PY_WIN" ]]; then
    echo "ERROR: Windows Python could not be resolved."
    exit 1
fi

PY_WSL="$(
    wslpath -u "$PY_WIN"
)"

echo
echo "Windows Python:"
echo "  $PY_WIN"

"$PY_WSL" -c '
import sys
print("platform:", sys.platform)
print("python:", sys.executable)
print("version:", sys.version)
'

# ------------------------------------------------------------
# Native Windows virtualenv
# ------------------------------------------------------------

echo
echo "[2/5] Creating Windows-native virtualenv..."

VENV_WIN="${REPO_WIN}\\.venv-win"
VENV_WSL="$REPO_WSL/.venv-win"

"$PY_WSL" \
    -m venv \
    "$VENV_WIN"

WINPY="$VENV_WSL/Scripts/python.exe"

if [[ ! -f "$WINPY" ]]; then
    echo "ERROR: Windows venv was not created:"
    echo "  $WINPY"
    exit 1
fi

"$WINPY" \
    -m pip install \
    --upgrade \
    pip \
    setuptools \
    wheel

# ------------------------------------------------------------
# PyTorch CUDA
#
# Match the successful Linux QLoRA runtime as closely as possible.
# ------------------------------------------------------------

echo
echo "[3/5] Installing Windows CUDA PyTorch..."

"$WINPY" \
    -m pip install \
    "torch==2.13.0+cu130" \
    --index-url \
    https://download.pytorch.org/whl/cu130

# ------------------------------------------------------------
# Training stack
# ------------------------------------------------------------

echo
echo "[4/5] Installing training stack..."

"$WINPY" \
    -m pip install \
    "transformers==5.16.1" \
    "accelerate==1.14.0" \
    "peft==0.20.0" \
    "bitsandbytes==0.50.2" \
    "datasets==4.8.5" \
    "trl==1.12.0" \
    "pyarrow>=25.0.1,<26"

"$WINPY" \
    -m pip install \
    -r "${REPO_WIN}\\requirements.txt"

# ------------------------------------------------------------
# Verification
# ------------------------------------------------------------

echo
echo "[5/5] Verifying native Windows CUDA..."

"$WINPY" - <<'PY'
import sys

import accelerate
import bitsandbytes
import peft
import torch
import transformers


print()
print("Native Windows CUDA Probe")
print("=========================")

print("platform:", sys.platform)
print("python:", sys.executable)

print()
print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if sys.platform != "win32":
    raise SystemExit(
        "ERROR: Python is not running as native Windows."
    )

if not torch.cuda.is_available():
    raise SystemExit(
        "ERROR: Windows-native PyTorch cannot see CUDA."
    )

index = torch.cuda.current_device()

free, total = torch.cuda.mem_get_info(
    index
)

print("GPU:", torch.cuda.get_device_name(index))
print(
    "BF16 supported:",
    torch.cuda.is_bf16_supported(),
)

print(
    f"VRAM free={free / 1024**3:.2f} GiB "
    f"total={total / 1024**3:.2f} GiB"
)

print()
print("bitsandbytes:", bitsandbytes.__version__)
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
print("peft:", peft.__version__)

print()
print("Running actual CUDA allocation...")

probe = torch.zeros(
    (1024, 1024),
    device="cuda",
)

torch.cuda.synchronize()

print(
    "CUDA allocation probe:",
    probe.device,
    tuple(probe.shape),
)

del probe

torch.cuda.empty_cache()

print()
print("NATIVE WINDOWS CUDA: PASS")
PY

echo
echo "============================================"
echo " Setup complete"
echo "============================================"
echo
echo "Windows venv:"
echo "  $VENV_WSL"
echo
echo "Use it directly from WSL:"
echo
echo "  WINPY=\"$WINPY\""
echo
echo '  "$WINPY" --version'
echo
echo '  "$WINPY" -c '\''import sys, torch; print(sys.platform, torch.cuda.is_available(), torch.cuda.get_device_name(0))'\'''
echo
echo "You do NOT need to activate the Windows venv."
echo "You do NOT need to open PowerShell."
echo
echo "IMPORTANT:"
echo "Do not launch the governed QLoRA trainer with it yet."
echo "Our immutable manifests currently contain /mnt/c/... paths."
echo "We will add the WSL <-> Windows path portability layer first."
