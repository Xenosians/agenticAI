"""One training worker for every Janus model: switch the job (model, dataset, stages), not the code.

This package is copied unchanged to Proxmox containers, remote GPUs and other machines. It may only
import the standard library, its own modules (relatively) and the training libraries listed in
``requirements-worker.txt``.
"""
