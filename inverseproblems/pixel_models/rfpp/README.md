# Instructions to solve Inverse-Problems using RF++

This is the codebase of RF++ paper [Improving the Training of Rectified Flows](https://arxiv.org/abs/2405.20320).

We provide extra inference scripts for the baselines and our work (FlowChef).

## Setup
1. Download the pretrained checkpoints from [RF++ release](https://drive.google.com/open?id=13cgGNkpOacb4HxlUM75ylcFOHFa0t2d1&usp=drive_fs):
   - `afhq-configF.pth` for AFHQ-Cat
   - `ffhq-configE.pth` for CelebA
   Place them in `./checkpoints/`

2. (optional) Download the test datasets:
   - AFHQ-Cat & CelebA test set from [dropbox](https://www.dropbox.com/scl/fo/3hr4qsh51st1tyxvegil7/AGn6W4piIIqSUlk9Pda79Uw?rlkey=nwu6lakmvdo8w7bd2apqhpnuj&st=wawqzll9&dl=0)
   Place them in `./data/AFHQ-Cat/` and `./data/CelebA/` respectively

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. The scripts support three inverse problems:
   - Box inpainting (mask_size: 20, 30)
   - Super resolution (scale_factor: 2, 4) 
   - Deblurring (kernel_size: 11, blur_sigma: 1.0, 5.0, 10.0)

   Each with optional noise (noise_sigma: 0.0, 0.05)


## Run inference

Run following scripts for all evaluations. 
Note that this only supports the AFHQ-Cat and CelebA.
We plan to release the imagenet weights soon to be able to do inference on diverse categories and upto 256x256.
Alternatively, look at our solutions for latent models.

> Note: These scripts assumes that 7 GPUs are available for parallel inference. Alternatively, modify the line 34.

```bash
# run FlowChef (ours) -- Gradient Free
bash ./inverseproblems_flowchef.sh

# run PnPFlow -- Gradient Free
bash ./inverseproblems_pnpflow.sh

# run DFlow
bash ./inverseproblems_dflow.sh

# run OTODE
bash ./inverseproblems_otode.sh

# run DPS
bash ./inverseproblems_dps.sh

# run FreeDoM
bash ./inverseproblems_freedom.sh
```

Results will be stored in `outputs/` with hyper-parameters.