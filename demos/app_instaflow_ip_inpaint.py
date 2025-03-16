import gradio as gr
import torch
from PIL import Image
import random
import numpy as np
import torch
import os
import json
from datetime import datetime

from src.pipeline_rf import RectifiedFlowPipeline

# Load the Stable Diffusion Inpainting model
pipe = RectifiedFlowPipeline.from_pretrained("XCLIU/2_rectified_flow_from_sd_1_5", torch_dtype=torch.float32)
pipe.to("cuda")  # Comment this line if GPU is not available

# Function to process the image
def process_image(
    image_layers, prompt, seed, randomize_seed, num_inference_steps,
    max_steps, learning_rate, optimization_steps, inverseproblem, mask_input
):
    image_with_mask = {
        "image": image_layers["background"],
        "mask": image_layers["layers"][0] if mask_input is None else mask_input
    }
    
    # Set seed
    if randomize_seed or seed is None:
        seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator("cuda").manual_seed(int(seed))

    # Unpack image and mask
    if image_with_mask is None:
        return None, f"❌ Please upload an image and create a mask."
    image = image_with_mask["image"]
    mask = image_with_mask["mask"]

    if image is None or mask is None:
        return None, f"❌ Please ensure both image and mask are provided."

    # Convert images to RGB
    image = image.convert("RGB")
    mask = mask.split()[-1]  # Convert mask to grayscale

    if not prompt:
        prompt = ""
        
    with torch.autocast("cuda"):
        # Placeholder for using advanced parameters in the future
        # Adjust parameters according to advanced settings if applicable
        result = pipe(
            prompt=prompt,
            negative_prompt="",
            input_image=image.resize((512, 512)),
            mask_image=mask.resize((512, 512)),
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,
            generator=generator,
            save_masked_image=False,
            output_path="test.png",
            learning_rate=learning_rate,
            max_steps=max_steps,
            optimization_steps=optimization_steps,
            inverseproblem=inverseproblem
        ).images[0]
    return result, f"✅ Inpainting completed with seed {seed}."

# Design the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(
        """
        <style>
            body {background-color: #f5f5f5; color: #333333;}
            h1 {text-align: center; font-family: 'Helvetica', sans-serif; margin-bottom: 10px;}
            h2 {text-align: center; color: #666666; font-weight: normal; margin-bottom: 30px;}
            .gradio-container {max-width: 800px; margin: auto;}
            .footer {text-align: center; margin-top: 20px; color: #999999; font-size: 12px;}
        </style>
        """
    )
    gr.Markdown("<h1>🍲 FlowChef 🍲</h1>")
    gr.Markdown("<h2>Inversion/Gradient/Training-free Steering of <u>InstaFlow (SDv1.5) for Inpainting (Inverse Problem)</u></h2>")
    gr.Markdown("<h3><p><a href='#'>Project Page</a> | <a href='#'>Paper</a></p> (Steering Rectified Flow Models in the Vector Field for Controlled Image Generation)</h3>")
    # gr.Markdown("<h3>💡 We recommend going through our <a href='#'>tutorial introduction</a> before getting started!</h3>")
    gr.Markdown("<h3>⚡ For better performance, check out our demo on <a href='#'>Flux</a>!</h3>")

    # Store current state
    current_input_image = None
    current_mask = None 
    current_output_image = None
    current_params = {}

    # Images at the top
    with gr.Row():
        with gr.Column():
            image_input = gr.ImageMask(
                # source="upload",
                # tool="sketch",
                type="pil",
                label="Input Image and Mask",
                image_mode="RGBA",
                height=600,
                width=600,
            )
        with gr.Column():
            output_image = gr.Image(label="Output Image")

    # All options below
    with gr.Column():
        prompt = gr.Textbox(
            label="Prompt",
            placeholder="Describe what should appear in the masked area..."
        )
        with gr.Row():
            seed = gr.Number(label="Seed (Optional)", value=None)
            randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
        num_inference_steps = gr.Slider(
            label="Inference Steps", minimum=50, maximum=200, value=100
        )
        # Advanced settings in an accordion
        with gr.Accordion("Advanced Settings", open=False):
            max_steps = gr.Slider(label="Max Steps", minimum=50, maximum=200, value=200)
            learning_rate = gr.Slider(label="Learning Rate", minimum=0.01, maximum=0.5, value=0.02)
            optimization_steps = gr.Slider(label="Optimization Steps", minimum=1, maximum=10, value=1)
            inverseproblem = gr.Checkbox(label="Apply mask on pixel space (works only when brush size is large)", value=False, info="Enables inverse problem formulation for inpainting by masking the RGB image itself. Hence, to avoid artifacts we increase the mask size manually during inference.")
            mask_input = gr.Image(
                type="pil",
                label="Optional Mask",
                image_mode="RGBA",
            )
        with gr.Row():
            run_button = gr.Button("Run", variant="primary")
            # save_button = gr.Button("Save Data", variant="secondary")

    # def update_visibility(selected_mode):
    #     if selected_mode == "Inpainting":
    #         return gr.update(visible=True), gr.update(visible=False)
    #     else:
    #         return gr.update(visible=True), gr.update(visible=True)

    # mode.change(
    #     update_visibility,
    #     inputs=mode,
    #     outputs=[prompt, edit_prompt],
    # )

    def run_and_update_status(
        image_with_mask, prompt, seed, randomize_seed, num_inference_steps,
        max_steps, learning_rate, optimization_steps, inverseproblem, mask_input
    ):
        result_image, result_status = process_image(
            image_with_mask, prompt, seed, randomize_seed, num_inference_steps,
            max_steps, learning_rate, optimization_steps, inverseproblem, mask_input
        )
        
        # Store current state
        global current_input_image, current_mask, current_output_image, current_params

        current_input_image = image_with_mask["background"] if image_with_mask else None
        current_mask = mask_input if mask_input is not None else (image_with_mask["layers"][0] if image_with_mask else None)
        current_output_image = result_image
        current_params = {
            "prompt": prompt,
            "seed": seed,
            "randomize_seed": randomize_seed,
            "num_inference_steps": num_inference_steps,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "optimization_steps": optimization_steps,
            "inverseproblem": inverseproblem,
        }
        
        return result_image

    def save_data():
        if not os.path.exists("saved_results"):
            os.makedirs("saved_results")
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join("saved_results", timestamp)
        os.makedirs(save_dir)
        
        # Save images
        if current_input_image:
            current_input_image.save(os.path.join(save_dir, "input.png"))
        if current_mask:
            current_mask.save(os.path.join(save_dir, "mask.png"))
        if current_output_image:
            current_output_image.save(os.path.join(save_dir, "output.png"))
            
        # Save parameters
        with open(os.path.join(save_dir, "parameters.json"), "w") as f:
            json.dump(current_params, f, indent=4)
            
        return f"✅ Data saved in {save_dir}"

    run_button.click(
        fn=run_and_update_status,
        inputs=[
            image_input,
            prompt,
            seed,
            randomize_seed,
            num_inference_steps,
            max_steps,
            learning_rate,
            optimization_steps,
            inverseproblem,
            mask_input
        ],
        outputs=output_image,
    )

    # save_button.click(fn=save_data)

    gr.Markdown(
        "<div class='footer'>Developed with ❤️ using InstaFlow (Stable Diffusion v1.5) and Gradio by Author.</div>"
    )

    def load_example_image_with_mask(image_path):
        # Load the image
        image = Image.open(image_path)
        # Create an empty mask of the same size
        mask = Image.new('L', image.size, 0)
        return {"background": image, "layers": [mask], "composite": image}

    # examples_dir = "assets"
    # volcano_dict = load_example_image_with_mask(os.path.join(examples_dir, "vulcano.jpg"))
    # dog_dict = load_example_image_with_mask(os.path.join(examples_dir, "dog.webp"))

    gr.Examples(
        examples=[
            [
                "./assets/saved_results/20241129_210517/input.png",  # image with mask
                "./assets/saved_results/20241129_210517/mask.png",
                "./assets/saved_results/20241129_210517/output.png",
                "a cat",  # prompt
                0,  # seed
                True,  # randomize_seed
                200,  # num_inference_steps
                200,  # max_steps
                0.1,  # learning_rate
                1,  # optimization_steps
                False,
            ],
            [
                "./assets/saved_results/20241129_211124/input.png",  # image with mask
                "./assets/saved_results/20241129_211124/mask.png",
                "./assets/saved_results/20241129_211124/output.png",
                " ",  # prompt
                0,  # seed
                True,  # randomize_seed
                200,  # num_inference_steps
                200,  # max_steps
                0.1,  # learning_rate
                5,  # optimization_steps
                False,
            ],
            [
                "./assets/saved_results/20241129_212001/input.png",  # image with mask
                "./assets/saved_results/20241129_212001/mask.png",
                "./assets/saved_results/20241129_212001/output.png",
                " ",  # prompt
                52,  # seed
                False,  # randomize_seed
                200,  # num_inference_steps
                200,  # max_steps
                0.02,  # learning_rate
                10,  # optimization_steps
                False,
            ],
            [
                "./assets/saved_results/20241129_212052/input.png",  # image with mask
                "./assets/saved_results/20241129_212052/mask.png",
                "./assets/saved_results/20241129_212052/output.png",
                " ",  # prompt
                52,  # seed
                False,  # randomize_seed
                200,  # num_inference_steps
                200,  # max_steps
                0.02,  # learning_rate
                10,  # optimization_steps
                False,
            ],
            [
                "./assets/saved_results/20241129_212155/input.png",  # image with mask
                "./assets/saved_results/20241129_212155/mask.png",
                "./assets/saved_results/20241129_212155/output.png",
                " ",  # prompt
                52,  # seed
                False,  # randomize_seed
                200,  # num_inference_steps
                200,  # max_steps
                0.02,  # learning_rate
                10,  # optimization_steps
                False,
            ],
        ],
        inputs=[
            image_input,
            mask_input,
            output_image,
            prompt,
            seed,
            randomize_seed,
            num_inference_steps,
            max_steps,
            learning_rate,
            optimization_steps,
            inverseproblem
        ],
        # outputs=[output_image],
        # fn=run_and_update_status,
        # cache_examples=True,
    )
demo.launch()