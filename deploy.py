"""Modal deploy entry for moss-image.

Deploy:
  modal deploy deploy.py

Design constraints:
  - Keep this file mostly self-contained because Modal remote imports may mount
    only the entry file.

MOSS-Image (OpenMOSS) is a 5B single-stream DiT with a Qwen3.5 text encoder and
the FLUX.2 small VAE decoder. It ships as a diffusers custom pipeline
(``pipeline_moss_image.py`` inside the weights repo, loaded with
``trust_remote_code``). Default weights: ``OpenMOSS-Team/MOSS-Image-SFT``; swap
``REPO_ID`` to ``OpenMOSS-Team/MOSS-Image-Base`` for the raw base model (same
architecture and sampling defaults). Prompts may be natural language, tag
sequences or a JSON layout (``summary`` + ``items`` with centred ``yxyx`` boxes).
"""

from __future__ import annotations

from pathlib import Path

import modal
from tongflow import deploy
from tongflow.models.image_gen import ImageGenInput, ImageGenOutput
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset
from tongflow.slots import current_params, node_slot


def _adv(name: str, default):
    """Advanced-section override (``TONGFLOW_SLOT_PARAMS``) or the plugin default."""
    v = current_params().get(name)
    if v is None:
        return default
    if isinstance(default, bool):
        return bool(v)
    if isinstance(default, int):
        return int(v)
    if isinstance(default, float):
        return float(v)
    return v

# Per-run knobs offered under the node's collapsed "Advanced" section.
# Pure literal (the platform scanner reads it by AST, never imports this
# module). Values reach the handlers via current_params(); an untouched
# control is absent there and falls back to the plugin default.
TONGFLOW_SLOT_PARAMS = {
    "image-gen": {
        "steps": {"type": "integer", "default": 50, "min": 10, "max": 100, "label": "Steps", "description": "Upstream recommends 50 or more."},
        "guidance_scale": {"type": "number", "default": 3.0, "min": 1.0, "max": 8.0, "step": 0.5, "label": "Guidance scale", "description": "3 suits long prompts; raise toward 5-6 for short ones."},
        "negative_prompt": {"type": "text", "default": "", "label": "Negative prompt", "description": "e.g. nsfw, worst quality, bad anatomy, watermark, jpeg artifact"},
        "skip_layer": {"type": "boolean", "default": True, "label": "Skip-layer guidance", "description": "Skip block 22 on the negative branch. Off looks more photographic but may lose detail."},
    },
}


REPO_ID = "OpenMOSS-Team/MOSS-Image-SFT"
MODEL_DIR = f"/models/{REPO_ID}"

DEFAULT_STEPS = 50
DEFAULT_GUIDANCE = 3.0
DEFAULT_NEGATIVE = ""
DEFAULT_SKIP_LAYER = True
UNCOND_SKIP_LAYERS = [22]

DEFAULT_SIZE = 1024
# Upstream trains around 1-2 MP and extrapolates from 0.25 MP to 4 MP.
MIN_SIDE = 256
MAX_SIDE = 2048

volume = modal.Volume.from_name("models", create_if_missing=True)


app = modal.App(Path(__file__).resolve().parent.name)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.9.1",
        "diffusers==0.38.0",
        "transformers==5.15.0",
        "accelerate==1.13.0",
        "safetensors==0.8.0",
        "pillow==12.1.1",
        "huggingface_hub>=1.5.0,<2.0",
        "tongflow==0.3.4", "fastapi[standard]",
    )
)

with image.imports():
    import torch
    from diffusers import DiffusionPipeline


def _side(v: int | None) -> int:
    """Clamp to the supported range and snap to a multiple of 16 (pipeline requirement)."""
    n = DEFAULT_SIZE if v is None else int(v)
    n = max(MIN_SIDE, min(MAX_SIDE, n))
    return max(16, round(n / 16) * 16)


@deploy
@app.cls(
    scaledown_window=2,
    image=image,
    gpu="H100",
    volumes={"/models": volume},
    timeout=1200,
)
class Inference:
    @modal.enter()
    def load(self):
        self.pipe = DiffusionPipeline.from_pretrained(
            MODEL_DIR,
            custom_pipeline=f"{MODEL_DIR}/pipeline_moss_image.py",
            trust_remote_code=True,
            torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
        )
        # Upstream re-instantiates the scheduler from its own config after load.
        self.pipe.scheduler = type(self.pipe.scheduler).from_config(self.pipe.scheduler.config)
        self.pipe.to("cuda")

    def _png_bytes(
        self,
        prompt: str,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        negative_prompt: str,
        skip_layer: bool,
        seed: int,
    ) -> bytes:
        import io

        result = self.pipe(
            prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            uncond_skip_layers=UNCOND_SKIP_LAYERS if skip_layer else [],
            # Upstream samples the initial noise on CPU in float32.
            generator=torch.Generator("cpu").manual_seed(seed),
        )
        buf = io.BytesIO()
        result.images[0].save(buf, format="PNG")
        return buf.getvalue()

    @modal.method()
    @node_slot(NodeSlots.IMAGE_GEN)
    def image_gen(self, input: ImageGenInput) -> ImageGenOutput:
        text = (input.text or "").strip()
        if not text:
            return ImageGenOutput(success=False, error="Missing text prompt")

        raw = self._png_bytes(
            text,
            height=_side(input.height),
            width=_side(input.width),
            num_inference_steps=_adv("steps", DEFAULT_STEPS),
            guidance_scale=_adv("guidance_scale", DEFAULT_GUIDANCE),
            negative_prompt=str(_adv("negative_prompt", DEFAULT_NEGATIVE) or ""),
            skip_layer=_adv("skip_layer", DEFAULT_SKIP_LAYER),
            seed=int(input.seed) if input.seed is not None else 42,
        )
        return ImageGenOutput(success=True, image=asset(raw, mime="image/png"))

    @modal.fastapi_endpoint(method="GET", label=f"{Path(__file__).resolve().parent.name}-serve")
    def serve(self, taskId: str = "", token: str = "", origin: str = ""):
        from fastapi.responses import StreamingResponse
        from tongflow import serve_stream_from_spec

        return StreamingResponse(
            serve_stream_from_spec(
                origin, taskId, token, __file__,
                invoke=lambda m, inp: getattr(self, m).local(inp),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"},
        )
