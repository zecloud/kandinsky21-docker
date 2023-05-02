import os
import argparse
import json
import torch
from PIL import Image
from torch import autocast


def read_multi_prompt(prompt):
    if prompt is None:
        return None
    with open(prompt, "r") as f:
        data=f.read()
        return json.loads(data)

def kandinsky_text2img(p):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from kandinsky2 import get_kandinsky2
    model = get_kandinsky2(p.device, task_type='text2img', model_version='2.1', use_flash_attention=False,cache_dir='/models/kandinsky2')

    if p.multi_prompt is not None:
        multiprompts=read_multi_prompt(p.multi_prompt)
        p.prompt=multiprompts[0]["prompt"]
        p.iters=len(multiprompts)
        prefix = p.prompt.replace(" ", "_")[:170]
    for j in range(p.iters):
        if p.multi_prompt is not None:
            p.prompt=multiprompts[j]["prompt"]
            prefix = p.prompt.replace(" ", "_")[:170]
        images = model.generate_text2img(
            p.prompt,
            num_steps=p.steps,
            batch_size=1,
            guidance_scale=p.scale,
            h=p.height,
            w=p.width,
            sampler=p.sampler, 
            prior_cf_scale=p.prior_scale,
            prior_steps=p.prior_steps
        )
        for i, img in enumerate(images):
            idx = j * p.samples + i + 1
            if(p.output_path is None and p.multi_prompt is None):
                out = f"{prefix}__steps_{p.steps}__scale_{p.scale:.2f}__n_{idx}.png"
                img.save(os.path.join("outputs", out))
            else:
                if p.multi_prompt is not None:
                    out=multiprompts[j]["output_path"]
                else:
                    out=p.output_path
                if(p.samples>1):
                    out=p.output_path.replace(".png","-"+str(j)+".png") 
                img.save(out)


def main():
    parser = argparse.ArgumentParser(description="Create images from a text prompt.")
    parser.add_argument(
            "--multi-prompt",
            type=str,
            nargs="?",
            help="A file with a list of prompts"
        )   
    parser.add_argument(
        "--output-path",
        type=str,
        nargs="?",
        help="Save the result in the output_path"
    )
    parser.add_argument(
        "--prompt", type=str, nargs="?", help="The prompt to render into an image"
    )
    parser.add_argument(
        "--image",
        type=str,
        nargs="?",
        help="The input image to use for image-to-image diffusion",
    )
    parser.add_argument(
        "--device",
        type=str,
        nargs="?",
        default="cuda",
        help="The cpu or cuda device to use to render images",
    )
    parser.add_argument(
        "--height", type=int, nargs="?", default=512, help="Image height in pixels"
    )
    parser.add_argument(
        "--width", type=int, nargs="?", default=512, help="Image width in pixels"
    )
    parser.add_argument(
        "--samples",
        type=int,
        nargs="?",
        default=1,
        help="Number of images to create per run",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        nargs="?",
        default="p_sampler",
        help="Override the sampler used to denoise the image",
    )
    parser.add_argument(
        "--iters",
        type=int,
        nargs="?",
        default=1,
        help="Number of times to run pipeline",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        nargs="?",
        help="The prompt to not render into an image",
    )
    parser.add_argument(
        "--scale",
        type=float,
        nargs="?",
        default=7.5,
        help="How closely the image should follow the prompt",
    )
    parser.add_argument(
        "--prior-scale",
        type=float,
        nargs="?",
        default=4,
        help="How closely the image should follow the prompt",
    )
    parser.add_argument(
        "--steps", type=int, nargs="?", default=50, help="Number of sampling steps"
    )
    parser.add_argument(
        "--prior-steps", type=str, nargs="?", default="25", help="Number of prior sampling steps"
    )                        
    args = parser.parse_args()
    kandinsky_text2img(args)

if __name__ == "__main__":
    main()