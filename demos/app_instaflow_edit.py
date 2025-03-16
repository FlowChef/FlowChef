import gradio as gr
import torch
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image
import random
import numpy as np
import torch
import os
import json
from datetime import datetime

from src.pipeline_rf_inversionfree_edit import RectifiedFlowPipeline as RectifiedFlowEditPipeline


pipe_edit = RectifiedFlowEditPipeline.from_pretrained("XCLIU/2_rectified_flow_from_sd_1_5", torch_dtype=torch.float32)
pipe_edit.to("cuda")

# Function to process the image
def process_image(
    image_layers, prompt, edit_prompt, seed, randomize_seed, num_inference_steps,
    max_steps, learning_rate, max_source_steps, optimization_steps, true_cfg, mask_input
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

    if not edit_prompt:
        return None, f"❌ Please provide a prompt for editing."
    if not prompt:
        prompt = ""
    # Resize the mask to match the image
    # mask = mask.resize(image.size)
    with torch.autocast("cuda"):
        # Placeholder for using advanced parameters in the future
        # Adjust parameters according to advanced settings if applicable
        result = pipe_edit(
            prompt=prompt,
            edit_prompt=edit_prompt,
            input_image=image.resize((512, 512)),
            mask_image=mask.resize((512, 512)),
            negative_prompt="",
            num_inference_steps=num_inference_steps,
            guidance_scale=true_cfg,
            generator=generator,
            # save_masked_image=False,
            # output_path="",
            learning_rate=learning_rate,
            max_steps=max_steps,
            optimization_steps=optimization_steps,
            full_source_steps=max_source_steps,
        ).images[0]
    return result, f"✅ Editing completed with seed {seed}."

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
    gr.Markdown("<h2>Inversion/Gradient/Training-free Steering of <u>InstaFlow (SDv1.5) for Image Editing</u></h2>")
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
        edit_prompt = gr.Textbox(
            label="Editing Prompt",
            placeholder="Describe how you want to edit the image..."
        )
        with gr.Row():
            seed = gr.Number(label="Seed (Optional)", value=None)
            randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
        num_inference_steps = gr.Slider(
            label="Inference Steps", minimum=10, maximum=100, value=50
        )
        # Advanced settings in an accordion
        with gr.Accordion("Advanced Settings", open=False):
            max_steps = gr.Slider(label="Max Steps", minimum=10, maximum=100, value=50)
            learning_rate = gr.Slider(label="Learning Rate", minimum=0.1, maximum=1.0, value=0.5)
            true_cfg = gr.Slider(label="Guidance Scale", minimum=1, maximum=20, value=2)
            max_source_steps = gr.Slider(label="Max Source Steps", minimum=1, maximum=200, value=40)
            optimization_steps = gr.Slider(label="Optimization Steps", minimum=1, maximum=10, value=1)
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
        image_with_mask, prompt, edit_prompt, seed, randomize_seed, num_inference_steps,
        max_steps, learning_rate, max_source_steps, optimization_steps, true_cfg, mask_input
    ):
        result_image, result_status = process_image(
            image_with_mask, prompt, edit_prompt, seed, randomize_seed, num_inference_steps,
            max_steps, learning_rate, max_source_steps, optimization_steps, true_cfg, mask_input
        )
        
        # Store current state
        global current_input_image, current_mask, current_output_image, current_params

        current_input_image = image_with_mask["background"] if image_with_mask else None
        current_mask = mask_input if mask_input is not None else (image_with_mask["layers"][0] if image_with_mask else None)
        current_output_image = result_image
        current_params = {
            "prompt": prompt,
            "edit_prompt": edit_prompt,
            "seed": seed,
            "randomize_seed": randomize_seed,
            "num_inference_steps": num_inference_steps,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "max_source_steps": max_source_steps,
            "optimization_steps": optimization_steps,
            "true_cfg": true_cfg,
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
            edit_prompt,
            seed,
            randomize_seed,
            num_inference_steps,
            max_steps,
            learning_rate,
            max_source_steps,
            optimization_steps,
            true_cfg,
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
                "./assets/saved_results/20241129_154837/input.png",  # image with mask
                "./assets/saved_results/20241129_154837/mask.png",
                "./assets/saved_results/20241129_154837/output.png",
                "a cat",  # prompt
                "a tiger",  # edit_prompt 
                0,  # seed
                True,  # randomize_seed
                50,  # num_inference_steps
                50,  # max_steps
                0.5,  # learning_rate
                20,  # max_source_steps
                5,  # optimization_steps
                2,  # true_cfg
            ],
            [
                "./assets/saved_results/20241129_195331/input.png",  # image with mask
                "./assets/saved_results/20241129_195331/mask.png",
                "./assets/saved_results/20241129_195331/output.png",
                "a cat",  # prompt
                "a silver sculpture of cat",  # edit_prompt 
                0,  # seed
                True,  # randomize_seed
                50,  # num_inference_steps
                50,  # max_steps
                0.5,  # learning_rate
                20,  # max_source_steps
                5,  # optimization_steps
                2,  # true_cfg
            ],
            [
                "./assets/saved_results/20241129_160439/input.png",  # image with mask
                "./assets/saved_results/20241129_160439/mask.png",
                "./assets/saved_results/20241129_160439/output.png",
                "a dog",  # prompt
                "a lion",  # edit_prompt 
                0,  # seed
                True,  # randomize_seed
                50,  # num_inference_steps
                20,  # max_steps
                0.5,  # learning_rate
                20,  # max_source_steps
                5,  # optimization_steps
                4,  # true_cfg
            ],
            [
                "./assets/saved_results/20241129_161118/input.png",  # image with mask
                "./assets/saved_results/20241129_161118/mask.png",
                "./assets/saved_results/20241129_161118/output.png",
                "two birds sitting on a branch",  # prompt
                "two origami birds sitting on a branch",  # edit_prompt 
                0,  # seed
                True,  # randomize_seed
                50,  # num_inference_steps
                50,  # max_steps
                0.5,  # learning_rate
                30,  # max_source_steps
                2,  # optimization_steps
                2,  # true_cfg
            ],
            [
                "./assets/saved_results/20241129_161602/input.png",  # image with mask
                "./assets/saved_results/20241129_161602/mask.png",
                "./assets/saved_results/20241129_161602/output.png",
                "a woman with long hair sitting in the sand at sunset",  # prompt
                "a woman with short hair sitting in the sand at sunset",  # edit_prompt 
                0,  # seed
                True,  # randomize_seed
                50,  # num_inference_steps
                30,  # max_steps
                0.5,  # learning_rate
                20,  # max_source_steps
                2,  # optimization_steps
                2,  # true_cfg
            ],
            [
                "./assets/saved_results/20241129_160150/input.png",  # image with mask
                "./assets/saved_results/20241129_160150/mask.png",
                "./assets/saved_results/20241129_160150/output.png",
                "A cute rabbit with big eyes",  # prompt
                "A cute pig with big eyes",  # edit_prompt 
                0,  # seed
                True,  # randomize_seed
                50,  # num_inference_steps
                40,  # max_steps
                0.5,  # learning_rate
                20,  # max_source_steps
                5,  # optimization_steps
                4,  # true_cfg
            ],
        ],
        inputs=[
            image_input,
            mask_input,
            output_image,
            prompt,
            edit_prompt,
            seed,
            randomize_seed,
            num_inference_steps,
            max_steps,
            learning_rate,
            max_source_steps,
            optimization_steps,
            true_cfg,
        ],
        # outputs=[output_image],
        # fn=run_and_update_status,
        # cache_examples=True,
    )
demo.launch()