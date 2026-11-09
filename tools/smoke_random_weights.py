"""CPU smoke test of the full online pipeline with random weights.

Checks that the model code still runs end to end on the installed versions of
torch, diffusers and transformers, without a GPU and without the real weights
(which come from Hugging Face). It builds every model PersonaLive loads with
random weights in WORK_DIR, then runs the online wrapper: fuse a reference
image, then animate one window of driving frames. The output is noise; the
test passes when every stage runs and returns frames of the expected shape.

Run it after every dependency change (tools/lock.sh), from the repo root:

    python tools/smoke_random_weights.py scratchpad/smoke

It needs about 8 GB of disk in WORK_DIR, 16 GB of RAM and a few minutes on CPU.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from diffusers import AutoencoderKL  # noqa: E402
from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection  # noqa: E402

from src.liveportrait.motion_extractor import MotionExtractor  # noqa: E402
from src.models.motion_encoder.encoder import MotEncoder  # noqa: E402
from src.models.pose_guider import PoseGuider  # noqa: E402
from src.models.unet_2d_condition import UNet2DConditionModel  # noqa: E402

# The base model's UNet (Stable Diffusion 1.x layout, as in
# lambda/sd-image-variations-diffusers), one layer per block to save time.
UNET_CONFIG = {
    "act_fn": "silu",
    "attention_head_dim": 8,
    "block_out_channels": [320, 640, 1280, 1280],
    "center_input_sample": False,
    "cross_attention_dim": 768,
    "down_block_types": ["CrossAttnDownBlock2D"] * 3 + ["DownBlock2D"],
    "downsample_padding": 1,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 4,
    "layers_per_block": 1,
    "mid_block_scale_factor": 1,
    "norm_eps": 1e-05,
    "norm_num_groups": 32,
    "out_channels": 4,
    "sample_size": 64,
    "up_block_types": ["UpBlock2D"] + ["CrossAttnUpBlock2D"] * 3,
}

VAE_CONFIG = {
    "act_fn": "silu",
    "block_out_channels": [128, 256, 512, 512],
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "in_channels": 3,
    "latent_channels": 4,
    "layers_per_block": 1,
    "norm_num_groups": 32,
    "out_channels": 3,
    "sample_size": 256,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
}


def build_weights(work: Path) -> Path:
    """Write random weights in the layout configs/prompts/*.yaml expects."""
    weights = work / "pretrained_weights"
    marker = weights / ".complete"
    if marker.exists():
        return weights
    torch.manual_seed(0)
    base = weights / "sd-image-variations-diffusers"
    UNet2DConditionModel(**UNET_CONFIG).save_pretrained(base / "unet")
    json.dump({"_class_name": "StableDiffusionImageVariationPipeline"},
              open(base / "model_index.json", "w"))
    vision = CLIPVisionConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                              num_attention_heads=2, image_size=224, patch_size=14,
                              projection_dim=768)
    CLIPVisionModelWithProjection(vision).save_pretrained(base / "image_encoder")
    AutoencoderKL(**VAE_CONFIG).save_pretrained(weights / "sd-vae-ft-mse")

    pl = weights / "personalive"
    pl.mkdir(parents=True, exist_ok=True)
    unet = UNet2DConditionModel.from_pretrained(base / "unet")
    torch.save(unet.state_dict(), pl / "reference_unet.pth")
    del unet
    torch.save(PoseGuider().state_dict(), pl / "pose_guider.pth")
    torch.save(MotEncoder().state_dict(), pl / "motion_encoder.pth")
    torch.save(MotionExtractor(num_kp=21).state_dict(), pl / "motion_extractor.pth")
    # Loaded with strict=False: the random initialisation stays.
    torch.save({}, pl / "denoising_unet.pth")
    torch.save({}, pl / "temporal_module.pth")
    marker.touch()
    return weights


def write_config(work: Path, weights: Path) -> Path:
    cfg = OmegaConf.load(REPO / "configs/prompts/personalive_online.yaml")
    cfg.dtype = "fp32"
    for key in list(cfg.keys()):
        value = cfg[key]
        if isinstance(value, str) and value.startswith("./pretrained_weights/"):
            cfg[key] = str(weights / value[len("./pretrained_weights/"):])
    cfg.inference_config = str(REPO / "configs/inference/inference_stage3.yaml")
    path = work / "smoke_online.yaml"
    OmegaConf.save(cfg, path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("work_dir", type=Path)
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    weights = build_weights(work)
    config_path = write_config(work, weights)
    print(f"weights ready in {time.time() - t0:.0f}s: {weights}")

    from src.wrapper import PersonaLive

    t0 = time.time()
    model = PersonaLive(SimpleNamespace(config_path=str(config_path), acceleration="none"),
                        device=torch.device("cpu"))
    print(f"models loaded in {time.time() - t0:.0f}s")

    rng = np.random.default_rng(0)
    ref = Image.fromarray(rng.integers(0, 255, (512, 512, 3), dtype=np.uint8))
    t0 = time.time()
    model.fuse_reference(ref)
    print(f"reference fused in {time.time() - t0:.0f}s")

    window = model.temporal_window_size
    frames = torch.rand(window, 3, 512, 512) * 2 - 1  # webcam frames in [-1, 1]
    t0 = time.time()
    video = model.process_input(frames)
    print(f"window animated in {time.time() - t0:.0f}s: output {video.shape}")
    expected = (window, 512, 512, 3)
    if tuple(video.shape) != expected or not np.isfinite(video).all():
        print(f"FAIL: expected finite frames of shape {expected}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
