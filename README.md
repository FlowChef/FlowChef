## 🚀 🚀 Steering Rectified Flow Models in the Vector Field for Controlled Image Generation

<div align="center">
  <a href="https://flowchef.github.io/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=GitHub&color=blue&logo=github"></a> &ensp;
  <a href="https://arxiv.org/abs/2412.00100"><img src="https://img.shields.io/static/v1?label=ArXiv&message=2412.00100&color=B31B1B&logo=arxiv"></a> &ensp;
  <a href="https://huggingface.co/spaces/FlowChef/FlowChef-Flux1-dev"><img src="https://img.shields.io/static/v1?label=Flux(editing x inverse problems)&message=Demo&color=yellow"></a> &ensp;
  <a href="https://huggingface.co/spaces/FlowChef/FlowChef-InstaFlow-Edit"><img src="https://img.shields.io/static/v1?label=InstaFlow(editing)&message=Demo&color=yellow"></a> &ensp;
  <a href="https://huggingface.co/spaces/FlowChef/FlowChef-InstaFlow-InverseProblem-Inpainting"><img src="https://img.shields.io/static/v1?label=InstaFlow(inpainting)&message=Demo&color=yellow"></a> &ensp;
</div>


<div align="center">
  <video src="https://github.com/user-attachments/assets/c0ac474d-726a-4003-8b9f-4ff32a400d82" autoplay muted loop playsinline style="width: 50%;"></video>
</div>


**FlowChef** introduces a novel approach to steer the rectified flow models (RFMs)  for controlled image generation for inversion-free, gradient-free, and training-free navigation of denoising trajectories. Unlike diffusion models, which demand extensive training and computational resources, **FlowChef** unifies tasks like **classifier guidance, inverse problems, and image editing** without extra training or backpropagation. **FlowChef** sets new benchmarks in performance, memory, and efficiency, achieving state-of-the-art results.

***We extend the FlowChef on Flux.1[dev] for image editing and inverse problems.*** 



## 🔥 Updates

- **[2025.03.15]** Scripts for gradio demo are provided in this repo.
- **[2024.11.29]**  All working demos are released on [HuggingFace](https://huggingface.co/FlowChef)!

## TODOs

- [ ] Extend the FlowChef to the video models.
- [ ] (top-priority) Release the support for [Inductive Moment Matching](https://github.com/lumalabs/imm) for inverse problems.
- [ ] (top-priority) Release the latent-space inverse problem benchmark script (with baselines).
- [ ] Release the diffusion baselines.
- ~~[x] Release the pixel-space inverse problem benchmark script (with baselines)~~
- ~~[x] Release the organized demo scripts~~
- ~~[x] Release the Flux.1[dev] demo~~
- ~~[x] Release the InstaFlow demo~~

## Instructions for `gradio_demos`

This folder contains all the gradio demos on Flux and InstaFLow. 
We provide Editing and Inpainting (Inverse Problem Setting) using FlowChef.

To set up the conda environment for the `gradio_demos` project, follow these steps:

```bash
# Clone the repository
# alternatively manually download the codebase
git clone https://github.com/FlowChef/FlowChef.git

cd FlowChef/demos

# Create a new conda environment
conda create --name flowchef_env python=3.10 -y

# Activate the conda environment
conda activate flowchef_env

# Install the required dependencies
pip install -r requirements.txt
```

By following these steps, you will have a conda environment set up and ready to run the `demos`.

### **Running the Gradio Demos**

Once the environment is set up, you can run the following demos:

1. **Flux Editing and Inpainting (Inverse Problem):**

    ```bash
    gradio app_flux.py
    ```

2. **InstaFlow Editing:**

    ```bash
    gradio app_instaflow_edit.py
    ```

3. **InstaFlow Inpainting (Inverse Problem):**

    ```bash
    gradio app_instaflow_ip_inpaint.py
    ```


## Instructions for `inverseproblems/pixel_models/rfpp`

This folder contains the code for solving inverse problems using RF++ models. We provide scripts for various baselines and our FlowChef approach.

For detailed instructions on:
- Setting up pretrained checkpoints
- Downloading test datasets 
- Running inference for different inverse problems (box inpainting, super resolution, deblurring)
- Evaluating baselines (DFlow, OTODE, DPS, FreeDoM) and FlowChef

Please refer to the detailed README in the [`inverseproblems/pixel_models/rfpp`](inverseproblems/pixel_models/rfpp/README.md) directory.

Note: The current implementation supports AFHQ-Cat and CelebA datasets. Support for ImageNet and higher resolutions (up to 256x256) will be released soon. Alternatively for these use cases, please check our latent model solutions.


## Citation
```
@article{patel2024flowchef,
        title={Steering Rectified Flow Models in the Vector Field for Controlled Image Generation},
        author={Patel, Maitreya and Wen, Song and Metaxas, Dimitris N. and Yang, Yezhou},
        journal={arXiv preprint arXiv:2412.00100},
        year={2024}
      }
```

## Ackowledgement

MP and YY are supported by NSF RI grants \#1750082 and \#2132724.  We thank the Research Computing (RC) at Arizona State University (ASU) and [cr8dl.ai](cr8dl.ai) for their generous support in providing computing resources. The views and opinions of the authors expressed herein do not necessarily state or reflect those of the funding agencies and employers.

We also acknowledge the authors of the Flux and InstaFlow models, along with HuggingFace, for releasing the models and source codes. 
