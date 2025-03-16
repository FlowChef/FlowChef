# From https://colab.research.google.com/drive/1LouqFBIC7pnubCOl5fhnFd33-oVJao2J?usp=sharing#scrollTo=yn1KM6WQ_7Em

import torch
import numpy as np
import torch.nn as nn
# import tensorboardX
import os
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from utils import straightness, parse_config, save_traj
import argparse
from tqdm import tqdm
from network_edm import SongUNet, DhariwalUNet, EDMPrecondVel
from torch.nn import DataParallel
import json
import matplotlib.pyplot as plt
from PIL import Image
import glob

from inverse_operators import *
import time

def get_args():
    parser = argparse.ArgumentParser(description='Configs')
    parser.add_argument('--gpu', type=str, help='gpu index')
    parser.add_argument('--dir', type=str, help='Saving directory name')
    parser.add_argument('--ckpt', type=str, default=None, help='Flow network checkpoint')
    parser.add_argument('--batchsize', type=int, default=1, help='Batch size')
    parser.add_argument('--N', type=int, default=20, help='Number of sampling steps')
    parser.add_argument('--save_traj', action='store_true', help='Save the trajectories')    
    parser.add_argument('--save_z', action='store_true', help='Save zs for distillation')    
    parser.add_argument('--save_data', action='store_true', help='Save data')    
    parser.add_argument('--solver', type=str, default='euler', help='ODE solvers')
    parser.add_argument('--config', type=str, default=None, help='Decoder config path, must be .json file')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--im_dir', type=str, help='Image dir')
    parser.add_argument('--action', type=str, default='sample', help='sample or interpolate')
    parser.add_argument('--compile', action='store_true', help='Compile the model')
    parser.add_argument('--t_steps', type=str, default=None, help='t_steps, e.g. 1,0.75,0.5,0.25')
    parser.add_argument('--sampler', type=str, default='default', help='default / new')

    # Inversion
    parser.add_argument('--data_path', type=str, default=None, help='Image path for inversion')
    parser.add_argument('--label_inv', type=int, help='Label for inversion')
    parser.add_argument('--label_rec', type=int, help='Label for reconstruction')    
    parser.add_argument('--N_decode', type=int, default=5, help='Number of decoding sampling steps')

    # Inverse problem arguments
    parser.add_argument('--inverse_problem', type=str, default='none', help='Type of inverse problem (deblur/box_inpaint/super_resolution)')
    parser.add_argument('--noise_sigma', type=float, default=0.0, help='Noise sigma for degradation')
    parser.add_argument('--kernel_size', type=int, default=11, help='Kernel size for deblurring')
    parser.add_argument('--blur_sigma', type=float, default=5.0, help='Blur sigma for deblurring')
    parser.add_argument('--mask_size', type=int, default=20, help='Box size for inpainting')
    parser.add_argument('--scale_factor', type=int, default=4, help='Scale factor for super resolution')
    parser.add_argument('--input_dir', type=str, help='Directory containing input images')

    parser.add_argument('--gradient_scale', type=float, default=100.0, help='Scale factor for gradients')
    parser.add_argument('--likebaseline', action='store_true', help='Like baseline disable mean loss')

    # pnp arguments
    parser.add_argument('--n_samples', type=float, default=5, help='PnP num_samples for averaging')

    return parser.parse_args()


@torch.no_grad()
def sample_ode_generative(model, z1, N, use_tqdm=True, solver='euler', label=None, inversion=False, 
                         time_schedule=None, sampler='default', operator=None, ref_img_path="", output_dir=""):
    assert solver in ['euler', 'heun']
    assert len(z1.shape) == 4
    assert operator is not None
    assert ref_img_path != ""

    pnp_steps = N
    n_samples = arg.n_samples

    downsample = operator.degradation
    upsample = operator.degradation_transpose

    if inversion:
        assert sampler == 'default'
    tq = tqdm if use_tqdm else lambda x: x

    if solver == 'heun' and N % 2 == 0:
        raise ValueError("N must be odd when using Heun's method.")
    if solver == 'heun':
        N = (N + 1) // 2
    traj = [] # to store the trajectory
    x0hat_list = []
    z1 = z1.detach()
    z = z1.clone()
    batchsize = z.shape[0]
    if time_schedule is not None:
        time_schedule = time_schedule + [0]
        sigma_schedule = [t_ / (1-t_ + 1e-6) for t_ in time_schedule]
        print(f"sigma_schedule: {sigma_schedule}")
    else:
        t_func = lambda i: i / N
        if inversion:
            time_schedule = [t_func(i) for i in range(0,N)] + [1]
            time_schedule[0] = 1e-3
        else:
            time_schedule = [t_func(i) for i in reversed(range(1,N+1))] + [0]
            time_schedule[0] = 1-1e-5

    cnt = 0

    # Load and preprocess image
    image = Image.open(ref_img_path).convert("RGB").resize((64, 64))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    preprocessed = transform(image).unsqueeze(0).to(z1.device)
    
    # Downsample and upsample
    downsampled = downsample(preprocessed)
    upsampled = upsample(downsampled)
    filename = os.path.splitext(os.path.basename(ref_img_path))[0]
    save_image(upsampled * 0.5 + 0.5, os.path.join(output_dir, f"{filename}_degraded.png"))

    
    config = model.module.config if hasattr(model, 'module') else model.config
    if config["label_dim"] > 0 and label is None:
        label = torch.randint(0, config["label_dim"], (batchsize,)).to(z1.device)
        label = F.one_hot(label, num_classes=config["label_dim"]).type(torch.float32)

    traj.append(z.detach().clone())

    z = upsampled.clone()

    # Track max GPU memory usage
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    for i in tq(range(pnp_steps)):
        time_i = torch.ones((batchsize), device=z.device) * time_schedule[i]
        with torch.enable_grad():
            z = z.detach() - (time_schedule[i]) * upsample(downsample(z) - downsampled) ## without bp

        z_samples = []
        for _ in range(n_samples):
            z_new = (1 - time_i) * z + torch.randn_like(z) * (time_i)
            z_sample = z_new - model(z_new, time_i, label) * (time_i)
            z_samples.append(z_sample)
        z = torch.stack(z_samples).mean(dim=0)

        x0hat_list.append(z.detach().clone())
        traj.append(z.detach().clone())

    end_time = time.time()
    max_memory = torch.cuda.max_memory_allocated() / (1024**2)  # Convert to MB
    total_time = end_time - start_time
    
    return traj, x0hat_list, max_memory, total_time

def main(arg):
    if not os.path.exists(arg.dir):
        os.makedirs(arg.dir)
    assert arg.config is not None
    config = parse_config(arg.config)
    arg.res = config['img_resolution']
    arg.input_nc = config['in_channels']
    arg.label_dim = config['label_dim']

    # Create directories
    samples_dir = os.path.join(arg.dir, "samples")
    if os.path.exists(samples_dir):
        num_files = len([f for f in os.listdir(samples_dir) if os.path.isfile(os.path.join(samples_dir, f))])
        if num_files >= 200:
            print(f"Directory {samples_dir} already contains {num_files} files. Exiting...")
            exit()
        
    os.makedirs(os.path.join(arg.dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(arg.dir, "zs"), exist_ok=True)
    os.makedirs(os.path.join(arg.dir, "trajs"), exist_ok=True)
    os.makedirs(os.path.join(arg.dir, 'tmp'), exist_ok=True)
    os.environ['TMPDIR'] = os.path.join(arg.dir, 'tmp')

    # Initialize model
    model_class = DhariwalUNet if config['unet_type'] == 'adm' else SongUNet
    flow_model = model_class(**config)

    # Setup device
    device_ids = arg.gpu.split(',')
    device = torch.device("cuda")
    print(f"Using {'multiple' if len(device_ids) > 1 else ''} GPU {arg.gpu}!")

    pytorch_total_params = sum(p.numel() for p in flow_model.parameters()) / 1000000
    print(f"Total parameters: {pytorch_total_params}M")

    flow_model = EDMPrecondVel(flow_model, use_fp16=config.get('use_fp16', False))

    if arg.ckpt is None:
        raise ValueError("Model checkpoint must be provided")
    flow_model.load_state_dict(torch.load(arg.ckpt, map_location="cpu"))
        
    if len(device_ids) > 1:
        flow_model = DataParallel(flow_model)
    flow_model = flow_model.to(device).eval()

    if arg.compile:
        flow_model = torch.compile(flow_model, mode="reduce-overhead", fullgraph=True)

    # Save configs
    with open(os.path.join(arg.dir, 'config_sampling.json'), 'w') as f:
        json.dump(vars(arg), f, indent=4)

    if arg.action == 'sample':
        sample(arg, flow_model, device)
    else:
        raise NotImplementedError(f"Action {arg.action} not implemented")


@torch.no_grad()
def sample(arg, model, device):
    output_dir = os.path.join(arg.dir, "samples")
    os.makedirs(output_dir, exist_ok=True)

    # Setup inverse problem
    sampling_config = SamplingConfig()
    sampling_config.inverse_problem = arg.inverse_problem
    sampling_config.noise_sigma = arg.noise_sigma
    sampling_config.kernel_size = arg.kernel_size
    sampling_config.blur_sigma = arg.blur_sigma
    sampling_config.mask_size = arg.mask_size
    sampling_config.scale_factor = arg.scale_factor

    operator = get_inverse_operator(sampling_config, device=device)

    input_images = sorted(glob.glob(os.path.join(arg.input_dir, "*.[pj][np][g]")))
    if not input_images:
        raise ValueError(f"No images found in {arg.input_dir}")
    print(f"Found {len(input_images)} input images")

    straightness_list = []
    nfes = []
    total_times = []
    max_memories = []

    for img_path in tqdm(input_images):
        z = torch.randn(arg.batchsize, arg.input_nc, arg.res, arg.res).to(device)
      
        if arg.label_dim > 0:
            label_onehot = torch.eye(arg.label_dim, device=device)[torch.randint(0, arg.label_dim, (z.shape[0],), device=device)]
        else:
            label_onehot = None

        if arg.solver in ['euler', 'heun']:
            t_steps = [float(t) for t in arg.t_steps.split(",")] if arg.t_steps else None
            if t_steps:
                t_steps[0] = 1-1e-5
            z = z * (1-1e-5)
            traj_uncond, traj_uncond_x0, max_memory, total_time = sample_ode_generative(
                model, z1=z, N=arg.N, use_tqdm=False, solver=arg.solver,
                label=label_onehot, time_schedule=t_steps, sampler=arg.sampler,
                operator=operator, ref_img_path=img_path, output_dir=output_dir
            )
            x0 = traj_uncond[-1]
            uncond_straightness = straightness(traj_uncond, mean=False)
            straightness_list.append(uncond_straightness)
            total_times.append(total_time)
            max_memories.append(max_memory)
        else:
            raise NotImplementedError(f"Solver {arg.solver} not implemented")

        if arg.save_traj:
            save_traj(traj_uncond, os.path.join(arg.dir, "trajs", f"{i:05d}_traj.png"))
            save_traj(traj_uncond_x0, os.path.join(arg.dir, "trajs", f"{i:05d}_traj_x0.png"))
            
        for idx in range(len(x0)):
            input_filename = os.path.basename(img_path)
            path_img = os.path.join(arg.dir, "samples", input_filename)
            path_z = os.path.join(arg.dir, "zs", input_filename.replace('.png', '.npy'))
            save_image(x0[idx] * 0.5 + 0.5, path_img)

            if arg.save_z:
                np.save(path_z, z[idx].cpu().numpy())

    # Save metrics
    straightness_list = torch.stack(straightness_list).view(-1).cpu().numpy()
    straightness_mean = np.mean(straightness_list).item()
    straightness_std = np.std(straightness_list).item()
    print(f"straightness.shape: {straightness_list.shape}")
    print(f"straightness_mean: {straightness_mean}, straightness_std: {straightness_std}")
    nfes_mean = np.mean(nfes) if len(nfes) > 0 else arg.N
    print(f"nfes_mean: {nfes_mean}")
    avg_time = np.mean(total_times)
    avg_memory = np.mean(max_memories)
    print(f"average_time: {avg_time:.2f}s")
    print(f"average_max_memory: {avg_memory:.2f}GB")
    
    result_dict = {
        "straightness_mean": straightness_mean,
        "straightness_std": straightness_std,
        "nfes_mean": nfes_mean,
        "average_time": avg_time,
        "average_max_memory": avg_memory
    }
    with open(os.path.join(arg.dir, 'result_sampling.json'), 'w') as f:
        json.dump(result_dict, f, indent=4)

    # Plot straightness distribution
    plt.figure()
    plt.hist(straightness_list, bins=20)
    plt.savefig(os.path.join(arg.dir, "straightness.png"))
    plt.close()


if __name__ == "__main__":
    arg = get_args()
    main(arg)