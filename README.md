# tongflow-modal-moss-image

Official [TongFlow](https://github.com/tong-io/tongflow) plugin. Text-to-image generation with **MOSS-Image** from [OpenMOSS](https://github.com/OpenMOSS/MOSS-Image) (`OpenMOSS-Team/MOSS-Image-SFT` by default), running on an H100 via [Modal](https://modal.com). An alternative to `tongflow-modal-z-image` on the same `image-gen` slot.

MOSS-Image is a 5B single-stream DiT with a Qwen3.5 text encoder. It is good at rendering dense Chinese and English text, accepts prompts up to 4K tokens, and understands JSON layout prompts that place text and objects by bounding box.

## Capabilities

- **Image generation** (`image-gen`): generate an image from a text prompt. Width and height are snapped to multiples of 16 and clamped to 256–2048; 1–2 MP (for example 1024×1024 or 1456×1456) works best.

### Prompts

- Natural language, tag sequences (`best quality, highres, 1girl, blue hair`) or both. English has the most training data; Chinese works well too.
- Put the exact text to render in quotes, and add line breaks where you want them.
- **Layout prompts (experimental):** paste a JSON object with `summary.caption` and `items` (`label`, `caption`, `bbox`, optional `text`), plus `"bbox_format": "yxyx"` and `"bbox_coord_space": "abs_centralized"` (pixel coordinates with the canvas centre as origin). Use the same width and height as the layout. See the [upstream user guide](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md#layout-prompts-experimental).

### Advanced

| Parameter | Default | Notes |
| --- | --- | --- |
| Steps | 50 | Upstream recommends 50 or more. |
| Guidance scale | 3.0 | 3 for long prompts; up to 5–6 for short ones. |
| Negative prompt | empty | Upstream suggests `nsfw, worst quality, bad anatomy, watermark, jpeg artifact`. |
| Skip-layer guidance | on | Skips block 22 on the negative branch. Off looks more photographic but may lose detail. |

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |

On first use the plugin downloads the weights (~27 GB) into the Modal volume `models`, deploys to your Modal account and caches the build. The weights are public; no Hugging Face token is needed.

## License

The plugin code and the MOSS-Image weights are both under the Apache License 2.0.
