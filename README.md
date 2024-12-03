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

- **[2024.11.29]**  All working demos are released on [HuggingFace](https://huggingface.co/FlowChef)!

## TODOs

- [x] Release the Flux.1[dev] demo
- [x] Release the InstaFlow demo
- [ ] Release the organized demo scripts
- [ ] Release the latent-space inverse problem benchmark script (with baselines)
- [ ] Release the pixel-space inverse problem benchmark script (with baselines)

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
