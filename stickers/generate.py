import torch
import numpy as np
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLControlNetImg2ImgPipeline, AutoencoderKL, ControlNetModel
from diffusers.utils import load_image 
from PIL import Image
import os
from transformers import DPTImageProcessor, DPTForDepthEstimation
import cv2
import argparse

parser = argparse.ArgumentParser(description="Run Stable Diffusion XL with LoRA.")
parser.add_argument("--use_controlnet", action='store_true', help="Don't use ControlNet when False")
parser.add_argument("--depth_control", action='store_true', help="Default control is canny; use depth when True")
parser.add_argument("--strength", type=float, default=0.3, help="Denoising strength (0.0-1.0)")
parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale")
parser.add_argument("--lora_path", type=str, default=None, help="LoRA model path")
parser.add_argument("--resolution", type=int, default=None, help="output resolution; 512 or 1024 recommended")
parser.add_argument("--input_path", type=str, default=None, help="Input path")
parser.add_argument("--output_path", type=str, default="outputs", help="Generated image output path")
parser.add_argument("--keyword", type=str, default="s3wnf3lt", help="Keyword for prompt")
args = parser.parse_args()

assert args.input_path is not None, "need to specify --input_path for argument"
assert args.lora_path is not None, "need to specify --lora_path for argument"

if args.use_controlnet:
    if args.depth_control:
        mode = "depth"
        depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
        feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
        control_model = "diffusers/controlnet-depth-sdxl-1.0"
    else:
        mode = "canny"
        control_model = "diffusers/controlnet-canny-sdxl-1.0"
    controlnet = ControlNetModel.from_pretrained(control_model, torch_dtype=torch.float16)

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            controlnet=controlnet,
            vae=vae,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16, 
            add_watermarker=False,
        )
else:
    mode = "nocontrol"
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, add_watermarker=False
        )

pipe = pipe.to("cuda")
pipe.load_lora_weights(pretrained_model_name_or_path_or_dict=args.lora_path)

def get_depth_map(image):
    size = image.size
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=size,
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)
    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image

def get_canny(image):
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image


dir_name = args.input_path
filenames = os.listdir(dir_name)
for file in filenames:
    name = file.split('.')[0]
    path = os.path.join(dir_name, file)
    input_image = Image.open(path).convert("RGB")
    if args.resolution:
        resol = (args.resolution, args.resolution)
    else:
        resol = input_image.size
    input_image = input_image.resize(resol, resample=Image.Resampling.LANCZOS)

    #prompt = f"{args.keyword} {name}"
    prompt = f"{args.keyword}"

    if args.depth_control:
        control_image = get_depth_map(input_image)
    else:
        control_image = get_canny(input_image)

    image = pipe(
        prompt,
        num_inference_steps=args.num_inference_steps,
        image=input_image,
        control_image=control_image,
        guidance_scale=args.guidance_scale, 
        strength=args.strength, 
        cross_attention_kwargs={"scale": 1.0}
    ).images[0]

    out_dir = f"{args.output_path}/outputs_str{args.strength}_cfg{args.guidance_scale}_{mode}"
    os.makedirs(out_dir, exist_ok=True)
    image.save(f"{out_dir}/{name}.png")

    del image
    torch.cuda.empty_cache()
