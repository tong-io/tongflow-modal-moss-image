"""Modal download entry for moss-image.

Run:
  modal run download.py::download

Self-contained: Modal remote execution may mount only this file, so do not
import other local modules.
"""

from __future__ import annotations

import os

import modal

# Keep in sync with deploy.py. MOSS-Image-Base shares the same layout (~27 GB).
REPO_ID = "OpenMOSS-Team/MOSS-Image-SFT"
MODEL_DIR = f"/models/{REPO_ID}"

volume = modal.Volume.from_name("models", create_if_missing=True)

model_downloader = modal.App("model_downloader")


@model_downloader.function(
    image=modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub==1.6.0",
    ),
    volumes={"/models": volume},
    timeout=3600,
)
def _download() -> None:
    from huggingface_hub import snapshot_download

    if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
        print(f"Model already exists at {MODEL_DIR}, skipping")
        return

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=MODEL_DIR,
        # README showcase images are not needed for inference.
        ignore_patterns=["assets/*"],
    )
    volume.commit()
    print(f"Model downloaded to {MODEL_DIR}")


@model_downloader.local_entrypoint()
def download() -> None:
    _download.remote()
