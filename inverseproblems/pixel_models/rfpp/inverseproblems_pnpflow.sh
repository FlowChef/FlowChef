#!/bin/bash

# Function to wait for available GPU
wait_for_gpu() {
    while true; do
        for gpu in "$@"; do
            # Check if GPU is both free and not locked
            if [ ! -f "/tmp/gpu_lock_${gpu}" ]; then
                # Get the list of PIDs using the GPU
                process_pids=$(nvidia-smi -i $gpu --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -E '^[0-9]+$')
                if [ -z "$process_pids" ]; then
                    # Create lock file
                    touch "/tmp/gpu_lock_${gpu}"
                    echo $gpu
                    return
                fi
            fi
        done
        sleep 10
    done
}

# Cleanup function to remove lock files
cleanup() {
    for gpu in "${GPUS[@]}"; do
        rm -f "/tmp/gpu_lock_${gpu}"
    done
}

# Set up trap to clean up lock files on script exit
trap cleanup EXIT

# Available GPUs
GPUS=(0 1 2 3 4 5 6)

# N=50 baseline experiments for AFHQ-Cat and CelebA in parallel
for dataset in "AFHQ-Cat" "CelebA"; do
    for problem in "box_inpaint" "super_resolution" "deblur"; do
        gpu=$(wait_for_gpu "${GPUS[@]}")

        # Set problem-specific parameters
        case $problem in
            "box_inpaint")
                params="--mask_size 20"
                ;;
            "super_resolution")
                params="--scale_factor 4"
                ;;
            "deblur")
                params="--kernel_size 11 --blur_sigma 1.0"
                ;;
        esac

        # Set dataset-specific checkpoints and configs
        if [ "$dataset" = "AFHQ-Cat" ]; then
            ckpt="afhq-configF.pth"
            config="afhq64_ve_aug.json"
        else
            ckpt="ffhq-configE.pth"
            config="ffhq64_ve_aug.json"
        fi

        (
            CUDA_VISIBLE_DEVICES=$gpu python generate_inverseproblems_pnpflow.py --gpu 0 \
                --solver euler --N 50 --sampler new --batchsize 1 \
                --ckpt ./checkpoints/${ckpt} \
                --config configs_unet/${config} \
                --input_dir ./data/${dataset} \
                --dir outputs/pnpflow/inverseproblems/N50_${dataset,,}_${problem} \
                --inverse_problem "${problem}" --noise_sigma 0.0 ${params} \
                --gradient_scale 500 ;
            rm -f "/tmp/gpu_lock_${gpu}"
        ) &
    done
done

# Wait for baseline experiments to complete
wait

# N=100 ablation experiments for both datasets in parallel
for dataset in "AFHQ-Cat" "CelebA"; do
    # Set dataset-specific checkpoints and configs
    if [ "$dataset" = "AFHQ-Cat" ]; then
        ckpt="afhq-configF.pth"
        config="afhq64_ve_aug.json"
    else
        ckpt="ffhq-configE.pth"
        config="ffhq64_ve_aug.json"
    fi

    # Box inpainting ablations
    for mask_size in 20 30; do
        for noise in 0.0 0.05; do
            gpu=$(wait_for_gpu "${GPUS[@]}")
            (
                CUDA_VISIBLE_DEVICES=$gpu python generate_inverseproblems_pnpflow.py --gpu 0 \
                    --solver euler --N 100 --sampler new --batchsize 1 \
                    --ckpt ./checkpoints/${ckpt} \
                    --config configs_unet/${config} \
                    --input_dir ./data/${dataset} \
                    --dir outputs/pnpflow/inverseproblems/N100_${dataset,,}_box_inpaint_m${mask_size}_n${noise} \
                    --inverse_problem "box_inpaint" --noise_sigma ${noise} \
                    --mask_size ${mask_size} --gradient_scale 500 ;
                rm -f "/tmp/gpu_lock_${gpu}"
            ) &
        done
    done

    # Super resolution ablations
    for scale in 2 4; do
        for noise in 0.0 0.05; do
            gpu=$(wait_for_gpu "${GPUS[@]}")
            (
                CUDA_VISIBLE_DEVICES=$gpu python generate_inverseproblems_pnpflow.py --gpu 0 \
                    --solver euler --N 100 --sampler new --batchsize 1 \
                    --ckpt ./checkpoints/${ckpt} \
                    --config configs_unet/${config} \
                    --input_dir ./data/${dataset} \
                    --dir outputs/pnpflow/inverseproblems/N100_${dataset,,}_super_resolution_s${scale}_n${noise} \
                    --inverse_problem "super_resolution" --noise_sigma ${noise} \
                    --scale_factor ${scale} --gradient_scale 500 ;
                rm -f "/tmp/gpu_lock_${gpu}"
            ) &
        done
    done

    # Deblur ablations
    for kernel in 1.0 5.0 10.0; do
        for noise in 0.0 0.05; do
            gpu=$(wait_for_gpu "${GPUS[@]}")
            (
                CUDA_VISIBLE_DEVICES=$gpu python generate_inverseproblems_pnpflow.py --gpu 0 \
                    --solver euler --N 100 --sampler new --batchsize 1 \
                    --ckpt ./checkpoints/${ckpt} \
                    --config configs_unet/${config} \
                    --input_dir ./data/${dataset} \
                    --dir outputs/pnpflow/inverseproblems/N100_${dataset,,}_deblur_k${kernel}_n${noise} \
                    --inverse_problem "deblur" --noise_sigma ${noise} \
                    --kernel_size 11 --blur_sigma ${kernel} --gradient_scale 500 ;
                rm -f "/tmp/gpu_lock_${gpu}"
            ) &
        done
    done
done

# Wait for all jobs to complete
wait

# # Test run with sample parameters
# CUDA_VISIBLE_DEVICES=0 python generate_inverseproblems_pnpflow_time.py --gpu 0 \
#     --solver euler --N 50 --sampler new --batchsize 1 \
#     --ckpt ./checkpoints/afhq-configF.pth \
#     --config configs_unet/afhq64_ve_aug.json \
#     --input_dir ./data/AFHQ-Cat \
#     --dir outputs/pnpflow/inverseproblems/test_run \
#     --inverse_problem "deblur" --noise_sigma 0.0 \
#     --kernel_size 11 --blur_sigma 5.0

