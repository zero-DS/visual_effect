import cv2
import torch
import numpy as np
from PIL import Image
from diffusers.utils import load_image
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
from transformers import CLIPVisionModelWithProjection
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import os
import argparse
import internvl3_utils
import re

parser = argparse.ArgumentParser(description="Run Stable Diffusion XL with ControlNet.")
parser.add_argument("--ip_dir_path", type=str, default=None, help="IP input dir path")
parser.add_argument("--mask_path", type=str, default=None, help="Mask input path")
parser.add_argument("--word", type=str, default=None, help="word for prompt")
parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
parser.add_argument("--resolution", type=int, default=1024, help="Resolution of output")
parser.add_argument("--control_scale", type=float, default=0.5, help="Controlnet conditioning scale")
parser.add_argument("--ip_scale", type=float, default=0.8, help="IP adapter conditioning scale")
parser.add_argument("--ip_styleonly", action='store_true', help="IP adapter conditioning method")
parser.add_argument("--use_captioning", action='store_true', help="Use captioning model for rich prompt describing IP")

args = parser.parse_args()

assert args.ip_dir_path is not None, "need to specify --ip_dir_path for argument"
assert args.mask_path is not None, "need to specify --mask_path for argument"
assert args.word is not None, "need to specify --word for argument"

captioning_prompt = """
Your objective is to identify the single most visually prominent subject in the image and describe ONLY its color, texture, and pattern.
Your entire response MUST be a single line of text, written exclusively in English, that strictly follows this exact template. Do not add any other words, labels, or line breaks.

Template:
Color: [Description], Texture: [Description], Pattern: [Description]

Crucial Rules:
1.  **English Only:** Your entire response must be in English.
2.  **Single Subject Only:** Focus exclusively on the one main subject. Do not describe anything else in the image, such as backgrounds or other objects.
3.  **Strict Template:** You must use the exact "Color:", "Texture:", and "Pattern:" labels. Do not introduce new categories.
4.  **Concise Descriptions:** Keep the description for each property brief and to the point.

Example:
For an image of a red brick wall, your output must be:
Color: Earthy red and muted grey, Texture: Rough and grainy, Pattern: Repetitive rectangular grid
"""

if args.ip_styleonly:
    stylemode = "_styleonly"
else:
    stylemode = ""
if args.use_captioning:
    captionmode = "_captioning"
else:
    captionmode = ""

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

if args.use_captioning:
    captioning_query = f"<image>\n{captioning_prompt}" 
    captioning_model = AutoModel.from_pretrained(
        'OpenGVLab/InternVL3-1B',
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVL3-1B', trust_remote_code=True, use_fast=False)


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

    if args.use_captioning:
        pixel_values = internvl3_utils.load_image(os.path.join(dirname, path), max_num=12).to(torch.bfloat16).cuda()
        generation_config = dict(max_new_tokens=1024, do_sample=True)
        response, _ = captioning_model.chat(tokenizer, pixel_values, captioning_query, generation_config, history=None, return_history=True)
        
        pattern = r"Color:\s*(.*?),\s*Texture:\s*(.*?),\s*Pattern:\s*(.*)"
        match = re.match(pattern, response)
        if match == None:
            prompt = f"Typography of '{args.word}'"
        else:
            color, texture, pattern = match.groups()
            prompt = f"Typography of '{args.word}', in {color} color, {texture} texture, {pattern} pattern."
    else:
        prompt = f"Typography of '{args.word}'"

    mask_image = load_image(args.mask_path)
    mask_image = mask_image.resize(resol)

    control_image = mask_image

        
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
    out_dir = f"results_ip{args.ip_scale}{stylemode}{captionmode}"
    os.makedirs(out_dir, exist_ok=True)
    image.save(f"{out_dir}/{name}.png")
