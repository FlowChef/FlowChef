import argparse
import torch
from PIL import Image
from src.fluxcombined import FluxPipeline
from src.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

def main():
    parser = argparse.ArgumentParser(description='Run FLUX image editing')
    parser.add_argument('--input_image', type=str, required=True, help='Path to input image')
    parser.add_argument('--mask_image', type=str, required=True, help='Path to mask image') 
    parser.add_argument('--output_path', type=str, default='outputs/flux_edit.png', help='Path to save output image')
    parser.add_argument('--prompt', type=str, required=True, help='Source prompt describing input image')
    parser.add_argument('--edit_prompt', type=str, required=True, help='Target prompt for editing')
    parser.add_argument('--num_inference_steps', type=int, default=30, help='Number of inference steps')
    parser.add_argument('--max_steps', type=int, default=30, help='Maximum number of steps')
    parser.add_argument('--learning_rate', type=float, default=0.6, help='Learning rate')
    parser.add_argument('--max_source_steps', type=int, default=10, help='Maximum source steps')
    parser.add_argument('--optimization_steps', type=int, default=4, help='Number of optimization steps')
    parser.add_argument('--true_cfg', type=float, default=4.5, help='True CFG value')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda/cpu)')
    
    args = parser.parse_args()

    # Set random seed
    generator = None#torch.Generator(device=args.device)#.manual_seed(args.seed)

    # Load images
    image = Image.open(args.input_image).convert("RGB")
    mask = Image.open(args.mask_image).split()[-1]  # Convert mask to grayscale by keeping alpha channel

    # Save mask image
    mask_output_path = args.output_path.rsplit('.', 1)[0] + '_mask.png'
    mask.save(mask_output_path)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained("black-forest-labs/FLUX.1-schnell", subfolder="scheduler")
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16, scheduler=scheduler)
    pipe.to(args.device)

    # Run edit
    result = pipe.edit(
        prompt=args.edit_prompt,
        input_image=image.resize((1024, 1024)),
        mask_image=mask.resize((1024, 1024)),
        num_inference_steps=args.num_inference_steps,
        guidance_scale=0.0,
        generator=generator,
        save_masked_image=False,
        output_path=args.output_path,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        optimization_steps=args.optimization_steps,
        true_cfg=args.true_cfg,
        negative_prompt=args.prompt,
        source_steps=args.max_source_steps,
    ).images[0]

    # Save result
    result.save(args.output_path)

if __name__ == "__main__":
    main()