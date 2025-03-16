import torch
import sys
import os
import argparse
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm
from src.pipeline_rf_inverseproblems import RectifiedFlowPipeline as RectifiedFlowPipelineInpaint

pipe = RectifiedFlowPipelineInpaint.from_pretrained("XCLIU/2_rectified_flow_from_sd_1_5", torch_dtype=torch.float32)
pipe.to("cuda")

def generate_image(prompt, input_image, operator, random_seed=False, learning_rate=0.02, max_steps=100, optimization_steps=1, num_inference_steps=100, guidance_scale=0.5, output_path=None, save_masked_image=False, *args, **kwargs):
    if random_seed:
        generator = None
    else:
        generator = torch.Generator(device="cuda").manual_seed(7984785)

    images = pipe(
        prompt=prompt,
        input_image=input_image,
        operator=operator,
        negative_prompt="",
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        max_steps=max_steps,
        optimization_steps=optimization_steps,
        learning_rate=learning_rate,
        save_masked_image=save_masked_image,
        output_path=output_path,
    ).images

    if output_path:
        images[0].save(output_path)

    return images[0]


if __name__ == "__main__":
    from inverse_operators import *
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Parent directory containing image folders")
    parser.add_argument("--output_dir", type=str, required=False, default="./outputs/", help="Output directory for generated images")
    parser.add_argument("--operation", type=str, required=True, choices=['inpaint', 'super', 'deblur'], 
                       help="Type of operation to perform")
    parser.add_argument("--box_size", type=int, default=128, help="Box size for inpainting")
    parser.add_argument("--scale_factor", type=int, default=4, help="Scale factor for super resolution")
    parser.add_argument("--kernel_size", type=int, default=50, help="Kernel size for deblurring")
    parser.add_argument("--sigma", type=float, default=5.0, help="Sigma for deblurring")
    parser.add_argument("--prompt", type=str, default="", help="Prompt for generation")

    # Add new arguments for generation parameters
    parser.add_argument("--random_seed", action="store_true", help="Use random seed instead of fixed seed")
    parser.add_argument("--learning_rate", type=float, default=0.02, help="Learning rate for optimization")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum number of optimization steps")
    parser.add_argument("--optimization_steps", type=int, default=1, help="Number of optimization steps per iteration")
    parser.add_argument("--num_inference_steps", type=int, default=100, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=0.5, help="Guidance scale for generation")

    args = parser.parse_args()

    # Set up operator based on args
    if args.operation == 'inpaint':
        operator = BoxInpaintingOperator(box_size=args.box_size)
    elif args.operation == 'super':
        operator = SuperResolutionOperator(scale_factor=args.scale_factor)
    elif args.operation == 'deblur':
        operator = GaussianDeblurOperator(kernel_size=args.kernel_size, sigma=args.sigma)

    # Create output directory
    output_dir = Path(args.output_dir) / args.operation
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all images in input directory
    input_dir = Path(args.input_dir)
    for image_path in tqdm(list(input_dir.rglob("*.jpg")) + list(input_dir.rglob("*.png"))):
        # Maintain folder structure in output
        rel_path = image_path.relative_to(input_dir)
        output_path = output_dir / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Set prompt based on input folder
        prompt = args.prompt

        if prompt=="":
            if "Cat" in str(image_path):
                prompt = "a cat"
            elif "Celeb" in str(image_path):
                prompt = "a person"
            elif "Style" in str(image_path):
                prompt = "a style image"
            else:
                print("No prompt provided -- using empty string as prompt.")

        # Process image
        input_image = Image.open(image_path).convert("RGB").resize((512, 512))
        generate_image(
            prompt=prompt,
            input_image=input_image,
            operator=operator,
            random_seed=args.random_seed,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            optimization_steps=args.optimization_steps,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            save_masked_image=True,
            output_path=str(output_path)
        )
