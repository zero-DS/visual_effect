import cv2
import torch
import numpy as np
from PIL import Image
from diffusers.utils import load_image
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
from transformers import CLIPVisionModelWithProjection
import os
import argparse

parser = argparse.ArgumentParser(description="Run Stable Diffusion XL with ControlNet.")
parser.add_argument("--ip_dir_path", type=str, default=None, help="IP input dir path")
parser.add_argument("--mask_path", type=str, default=None, help="Mask input path")
parser.add_argument("--word", type=str, default=None, help="word for prompt")
parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
parser.add_argument("--resolution", type=int, default=1024, help="Resolution of output")
parser.add_argument("--control_scale", type=float, default=0.5, help="Controlnet conditioning scale")
parser.add_argument("--ip_scale", type=float, default=0.8, help="IP adapter conditioning scale")
parser.add_argument("--ip_styleonly", action='store_true', help="IP adapter conditioning method")

args = parser.parse_args()

assert args.ip_dir_path is not None, "need to specify --ip_dir_path for argument"
assert args.mask_path is not None, "need to specify --mask_path for argument"
assert args.word is not None, "need to specify --word for argument"

if args.ip_styleonly:
    stylemode = "styleonly"
else:
    stylemode = ""

dirname = args.ip_dir_path
paths = os.listdir(dirname)
for path in paths:
    name = path.split('.')[0]

    ip_image = Image.open(os.path.join(dirname, path)).convert("RGB")

    resol = (args.resolution, args.resolution)

    init_image = load_image(
            "transparent.png"
            #"white.png"
    )
    init_image = init_image.resize(resol)

    mask_image = load_image(args.mask_path)
    mask_image = mask_image.resize(resol)

    control_image = mask_image

    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter", subfolder="models/image_encoder"
    ).to(device="cuda", dtype=torch.float16)

    pipeline = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", image_encoder=image_encoder, controlnet=controlnet, torch_dtype=torch.float16
    ).to(device="cuda")

    pipeline.load_ip_adapter(
      "h94/IP-Adapter",
      subfolder="sdxl_models",
      weight_name="ip-adapter-plus_sdxl_vit-h.safetensors"
    )
    if args.ip_styleonly:
        ip_scale = {
        "up": {"block_0": [0.0, args.ip_scale, 0.0]},
        }
    else:
        ip_scale = args.ip_scale

    pipeline.set_ip_adapter_scale(ip_scale)

    prompt = f"Typography of '{args.word}'"

    image = pipeline(
        prompt=prompt,
        num_inference_steps=args.num_inference_steps,
        strength=0.99,
        controlnet_conditioning_scale=args.control_scale,
        image=init_image,
        mask_image=mask_image,
        ip_adapter_image=ip_image,
        control_image=mask_image,
    ).images[0]
    out_dir = f"results_ip{args.ip_scale}_{stylemode}"
    os.makedirs(out_dir, exist_ok=True)
    image.save(f"{out_dir}/{name}.png")
