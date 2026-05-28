# HidraChat Worker — Windows start script
# Run in PowerShell: .\start.ps1

Write-Host "⚡ HidraChat Worker" -ForegroundColor Cyan

# Optional: set env vars here
# $env:HIDRACHAT_WORKER_NAME  = "my-gpu-worker"
# $env:HIDRACHAT_WORKER_EMAIL = "you@email.com"
# $env:HIDRACHAT_N_GPU_LAYERS = "35"   # set > 0 to use GPU (CUDA/Vulkan)
# $env:HIDRACHAT_THREADS      = "8"
# $env:HIDRACHAT_RAM_GB       = "8"

python worker.py
