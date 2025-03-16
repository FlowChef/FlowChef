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

from src.fluxcombined import FluxPipeline
from src.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

# Load the Stable Diffusion Inpainting model
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="scheduler")
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16, scheduler=scheduler)
pipe.to("cuda")  # Comment this line if GPU is not available

# Function to process the image
def process_image(
    mode, image_layers, prompt, edit_prompt, seed, randomize_seed, num_inference_steps,
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

    if mode == "Inpainting":
        if not prompt:
            return None, f"❌ Please provide a prompt for inpainting."
        with torch.autocast("cuda"):
            # height = ((image.size[1] + 7) // 8) * 8
            # width = ((image.size[0] + 7) // 8) * 8
            # image = image.resize((width, height))
            # mask = mask.resize((width, height))

            result = pipe.inpaint(
                prompt=prompt,
                input_image=image.resize((1024, 1024)),
                mask_image=mask.resize((1024, 1024)),
                num_inference_steps=num_inference_steps,
                guidance_scale=0.5,
                generator=generator,
                save_masked_image=False,
                output_path="",
                learning_rate=learning_rate,
                max_steps=max_steps,
                optimization_steps=optimization_steps,
                # height=image.size[1],
                # width=image.size[0],
            ).images[0]
            pipe.vae = pipe.vae.to(torch.float16)
            torch.cuda.empty_cache()
            
        return result, f"✅ Inpainting completed with seed {seed}."
    elif mode == "Editing":
        if not edit_prompt:
            return None, f"❌ Please provide a prompt for editing."
        if not prompt:
            prompt = ""
        # Resize the mask to match the image
        # mask = mask.resize(image.size)
        with torch.autocast("cuda"):
            # height = ((image.size[1] + 7) // 8) * 8
            # width = ((image.size[0] + 7) // 8) * 8
            # image = image.resize((width, height))
            # mask = mask.resize((width, height))

            result = pipe.edit(
                prompt=edit_prompt,
                input_image=image.resize((1024, 1024)),
                mask_image=mask.resize((1024, 1024)),
                num_inference_steps=num_inference_steps,
                guidance_scale=0.0,
                generator=generator,
                save_masked_image=False,
                output_path="",
                learning_rate=learning_rate,
                max_steps=max_steps,
                optimization_steps=optimization_steps,
                true_cfg=true_cfg,
                negative_prompt=prompt,
                source_steps=max_source_steps,
                # height=image.size[1],
                # width=image.size[0],
            ).images[0]
            torch.cuda.empty_cache()
            
        return result, f"✅ Editing completed with seed {seed}."
    else:
        return None, f"❌ Invalid mode selected."

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
    gr.Markdown("<h2>Inversion/Gradient/Training-free Steering of <u>Flux.1[dev]</u></h2>")
    gr.Markdown("<h2><p><a href='#'>Project Page</a> | <a href='#'>Paper</a></p> (Steering Rectified Flow Models in the Vector Field for Controlled Image Generation)</h2>")
    # gr.Markdown("<h3>💡 We recommend going through our <a href='#'>tutorial introduction</a> before getting started!</h3>")
    gr.Markdown("<h3>⚡ For faster generation, check out our demo on InstaFlow <a href='#'>Editing</a> and <a href='#'>Inpainting</a>!</h3>")

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
        mode = gr.Radio(
            choices=["Inpainting", "Editing"], label="Select Mode", value="Inpainting"
        )
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
            label="Inference Steps", minimum=1, maximum=50, value=30
        )
        # Advanced settings in an accordion
        with gr.Accordion("Advanced Settings", open=False):
            max_steps = gr.Slider(label="Max Steps", minimum=1, maximum=30, value=30)
            learning_rate = gr.Slider(label="Learning Rate", minimum=0.1, maximum=1.0, value=0.5)
            true_cfg = gr.Slider(label="Guidance Scale (only for editing)", minimum=1, maximum=20, value=2)
            max_source_steps = gr.Slider(label="Max Source Steps (only for editing)", minimum=1, maximum=30, value=20)
            optimization_steps = gr.Slider(label="Optimization Steps", minimum=1, maximum=10, value=1)
            mask_input = gr.Image(
                type="pil",
                label="Optional Mask",
                image_mode="RGBA",
            )
        with gr.Row():
            run_button = gr.Button("Run", variant="primary")
            # save_button = gr.Button("Save Data", variant="secondary")

    def update_visibility(selected_mode):
        if selected_mode == "Inpainting":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=True), gr.update(visible=True)

    mode.change(
        update_visibility,
        inputs=mode,
        outputs=[prompt, edit_prompt],
    )

    def run_and_update_status(
        mode, image_with_mask, prompt, edit_prompt, seed, randomize_seed, num_inference_steps,
        max_steps, learning_rate, max_source_steps, optimization_steps, true_cfg, mask_input
    ):
        result_image, result_status = process_image(
            mode, image_with_mask, prompt, edit_prompt, seed, randomize_seed, num_inference_steps,
            max_steps, learning_rate, max_source_steps, optimization_steps, true_cfg, mask_input
        )
        
        # Store current state
        global current_input_image, current_mask, current_output_image, current_params

        current_input_image = image_with_mask["background"] if image_with_mask else None
        current_mask = mask_input if mask_input is not None else (image_with_mask["layers"][0] if image_with_mask else None)
        current_output_image = result_image
        current_params = {
            "mode": mode,
            "prompt": prompt,
            "edit_prompt": edit_prompt,
            "seed": seed,
            "randomize_seed": randomize_seed,
            "num_inference_steps": num_inference_steps,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "max_source_steps": max_source_steps,
            "optimization_steps": optimization_steps,
            "true_cfg": true_cfg
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
            mode,
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
        "<div class='footer'>Developed with ❤️ using Flux and Gradio by Author.</div>"
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
                "Inpainting",  # mode
                "./assets/saved_results/20241126_053639/input.png",  # image with mask
                "./assets/saved_results/20241126_053639/mask.png",
                "./assets/saved_results/20241126_053639/output.png",
                "a dog",  # prompt
                " ",  # edit_prompt 
                0,  # seed
                True,  # randomize_seed
                30,  # num_inference_steps
                30,  # max_steps
                1.0,  # learning_rate
                20,  # max_source_steps
                5,  # optimization_steps
                2,  # true_cfg
            ],
            [
                "Inpainting",  # mode
                "./assets/saved_results/20241126_173140/input.png",  # image with mask
                "./assets/saved_results/20241126_173140/mask.png",
                "./assets/saved_results/20241126_173140/output.png",
                "a cat with blue eyes",  # prompt
                " ",  # edit_prompt 
                0,  # seed
                True,  # randomize_seed
                30,  # num_inference_steps
                20,  # max_steps
                1.0,  # learning_rate
                20,  # max_source_steps
                5,  # optimization_steps
                2,  # true_cfg
            ],
            [
                "Editing",  # mode
                "./assets/saved_results/20241126_181633/input.png",  # image with mask
                "./assets/saved_results/20241126_181633/mask.png",
                "./assets/saved_results/20241126_181633/output.png",
                " ",  # prompt
                "volcano eruption",  # edit_prompt 
                0,  # seed
                True,  # randomize_seed
                30,  # num_inference_steps
                20,  # max_steps
                0.4,  # learning_rate
                4,  # max_source_steps
                5,  # optimization_steps
                4.5,  # true_cfg
            ],
            [
                "Editing",  # mode
                "./assets/saved_results/20241126_214810/input.png",  # image with mask
                "./assets/saved_results/20241126_214810/mask.png",
                "./assets/saved_results/20241126_214810/output.png",
                " ",  # prompt
                "a dog with flowers in the mouth",  # edit_prompt 
                0,  # seed
                True,  # randomize_seed
                30,  # num_inference_steps
                30,  # max_steps
                1,  # learning_rate
                5,  # max_source_steps
                3,  # optimization_steps
                4.5,  # true_cfg
            ],
            [
                "Inpainting",  # mode
                "./assets/saved_results/20241127_025429/input.png",  # image with mask
                "./assets/saved_results/20241127_025429/mask.png",
                "./assets/saved_results/20241127_025429/output.png",
                "A building with \"ASU\" written on it.",  # prompt
                "",  # edit_prompt 
                52,  # seed
                False,  # randomize_seed
                30,  # num_inference_steps
                30,  # max_steps
                1,  # learning_rate
                20,  # max_source_steps
                3,  # optimization_steps
                2,  # true_cfg
            ],
            [
                "Inpainting",  # mode
                "./assets/saved_results/20241126_222257/input.png",  # image with mask
                "./assets/saved_results/20241126_222257/mask.png",
                "./assets/saved_results/20241126_222257/output.png",
                "A cute pig with big eyes",  # prompt
                "",  # edit_prompt 
                0,  # seed
                True,  # randomize_seed
                30,  # num_inference_steps
                20,  # max_steps
                1,  # learning_rate
                20,  # max_source_steps
                3,  # optimization_steps
                2,  # true_cfg
            ],
            [
                "Editing",  # mode
                "./assets/saved_results/20241126_222522/input.png",  # image with mask
                "./assets/saved_results/20241126_222522/mask.png",
                "./assets/saved_results/20241126_222522/output.png",
                "A cute rabbit with big eyes",  # prompt
                "A cute pig with big eyes",  # edit_prompt 
                0,  # seed
                True,  # randomize_seed
                30,  # num_inference_steps
                20,  # max_steps
                0.4,  # learning_rate
                5,  # max_source_steps
                5,  # optimization_steps
                4.5,  # true_cfg
            ],
            [
                "Editing",  # mode
                "./assets/saved_results/20241126_223719/input.png",  # image with mask
                "./assets/saved_results/20241126_223719/mask.png",
                "./assets/saved_results/20241126_223719/output.png",
                "a cat",  # prompt
                "a tiger",  # edit_prompt 
                0,  # seed
                True,  # randomize_seed
                30,  # num_inference_steps
                30,  # max_steps
                0.6,  # learning_rate
                10,  # max_source_steps
                4,  # optimization_steps
                4.5,  # true_cfg
            ],
        ],
        inputs=[
            mode,
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