import argparse
import torch
from PIL import Image
from src.pipeline_rf_inversionfree_edit import RectifiedFlowPipeline as RectifiedFlowEditPipeline

def main():
    parser = argparse.ArgumentParser(description='Run InstaFlow image editing')
    parser.add_argument('--input_image', type=str, required=True, help='Path to input image')
    parser.add_argument('--mask_image', type=str, required=True, help='Path to mask image')
    parser.add_argument('--output_path', type=str, default='outputs/instaflow_edit.png', help='Path to save output image')
    parser.add_argument('--prompt', type=str, required=True, help='Source prompt describing input image')
    parser.add_argument('--edit_prompt', type=str, required=True, help='Target prompt for editing')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--max_steps', type=int, default=50, help='Maximum number of steps')
    parser.add_argument('--learning_rate', type=float, default=0.5, help='Learning rate')
    parser.add_argument('--max_source_steps', type=int, default=20, help='Maximum source steps')
    parser.add_argument('--optimization_steps', type=int, default=5, help='Number of optimization steps')
    parser.add_argument('--true_cfg', type=float, default=2.0, help='Guidance scale value')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda/cpu)')

    args = parser.parse_args()

    # Set random seed
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    # Load images
    image = Image.open(args.input_image).convert("RGB")
    mask = Image.open(args.mask_image).split()[-1]  # Convert mask to grayscale

    # Save mask image
    mask_output_path = args.output_path.rsplit('.', 1)[0] + '_mask.png'
    mask.save(mask_output_path)

    # Initialize pipeline
    pipe_edit = RectifiedFlowEditPipeline.from_pretrained(
        "XCLIU/2_rectified_flow_from_sd_1_5", 
        torch_dtype=torch.float32
    )
    pipe_edit.to(args.device)

    # Run edit
    result = pipe_edit(
        prompt=args.prompt,
        edit_prompt=args.edit_prompt,
        input_image=image.resize((512, 512)),
        mask_image=mask.resize((512, 512)),
        negative_prompt="",
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.true_cfg,
        generator=generator,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        optimization_steps=args.optimization_steps,
        full_source_steps=args.max_source_steps,
    ).images[0]

    # Save result
    result.save(args.output_path)

if __name__ == "__main__":
    main()