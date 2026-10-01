#!/usr/bin/env bash
set -euo pipefail

export WINPY="$PWD/.venv-win/Scripts/python.exe"

if [[ ! -f "$WINPY" ]]; then
    echo "ERROR: Windows Python not found:"
    echo "  $WINPY"
    return 1 2>/dev/null || exit 1
fi

# Useful while diagnosing/training the native Windows CUDA path.
export CUDA_LAUNCH_BLOCKING=1

# Explicitly expose this variable to Win32 processes launched from WSL.
case ":${WSLENV:-}:" in
    *":CUDA_LAUNCH_BLOCKING:"*)
        ;;
    *)
        export WSLENV="${WSLENV:+${WSLENV}:}CUDA_LAUNCH_BLOCKING"
        ;;
esac

# Do not propagate the Linux allocator option to native Windows.
unset PYTORCH_CUDA_ALLOC_CONF

echo "Native Windows training environment"
echo "==================================="
echo "WINPY=$WINPY"
echo "CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"
echo "WSLENV=$WSLENV"
