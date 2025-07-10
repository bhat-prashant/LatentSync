#!/bin/bash

# Start the timer
start_time=$(date +%s)

# Run your Python script
python -m scripts.inference \
    --unet_config_path "configs/unet/stage2_512.yaml" \
    --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
    --inference_steps 20 \
    --guidance_scale 1.5 \
    --enable_deepcache \
    --image_enhance \
    --video_path "assets/demo1.mp4" \
    --audio_path "assets/demo3_audio.wav" \
    --video_out_path "out1_en.mp4"

# End the timer
end_time=$(date +%s)

# Calculate execution time
execution_time=$((end_time - start_time))

# Convert execution time to minutes and seconds
minutes=$((execution_time / 60))
seconds=$((execution_time % 60))

# Print execution time
echo "Total execution time: ${minutes} minutes and ${seconds} seconds"
