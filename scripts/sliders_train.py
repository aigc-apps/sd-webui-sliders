import os
import platform
import subprocess
import sys
from shutil import copyfile

from scripts.sliders_config import (
    cache_log_file_path,
    sliders_models_path,
    models_path,
    sliders_id_outpath_samples
)
from scripts.sliders_utils import check_sliders_valid, check_files_exists_and_download
from scripts.sdwebui import get_checkpoint_type, unload_sd
from PIL import Image, ImageOps

python_executable_path = sys.executable
check_hash = [True, True]


# Attention! Output of js is str or list, not float or int
@unload_sd()
def sliders_train_forward(
    sd_model_checkpoint: str,
    id_task: str,
    low_instance_images: list,
    high_instance_images: list,
    sliders_id: str,
    train_mode_choose: str,
    target_prompt: str, 
    positive_prompt: str,
    neutral_prompt: str,
    unconditional_prompt: str,
    resolution: int,
    checkpointing_steps: int,
    max_train_steps: int,
    train_batch_size: int,
    learning_rate: float,
    rank: int,
    network_alpha: int,
    *args,
):
    global check_hash

    if sliders_id == "" or sliders_id is None:
        return "Feature edit id cannot be set to empty."
    if sliders_id == "none":
        return "Feature edit id cannot be set to none."

    ids = []
    _sliders_ids = os.listdir(os.path.join(models_path, "Lora"))
    for _sliders_id in _sliders_ids:
        if check_sliders_valid(os.path.join(models_path, "Lora", _sliders_id)):
            ids.append(os.path.splitext(_sliders_id)[0])
    ids = sorted(ids)

    if sliders_id in ids:
        return "Feature edit id non-repeatability."

    if int(rank) < int(network_alpha):
        return "The network alpha {} must not exceed rank {}. " "It will result in an unintended LoRA.".format(network_alpha, rank)

    check_files_exists_and_download(check_hash[0], "base")
    check_hash[0] = False

    checkpoint_type = get_checkpoint_type(sd_model_checkpoint)
    if checkpoint_type == 2 or checkpoint_type==3:
        return "sd-webui-wliders does not support the SD2 checkpoint or SDXL checkpoint: {}.".format(sd_model_checkpoint)
    sdxl_pipeline_flag = True if checkpoint_type == 3 else False

    # Training weight saving
    weights_save_path = os.path.join(sliders_id_outpath_samples, sliders_id, "user_weights")
    # Raw data backup
    sliders_path = os.path.join(sliders_id_outpath_samples, sliders_id, "processed_images")
    low_data_backup_path = os.path.join(sliders_path, "low")
    high_data_backup_path = os.path.join(sliders_path, "high")
    
    webui_save_path = os.path.join(models_path, f"Lora/sliders_{sliders_id}.safetensors")
    webui_load_path = os.path.join(models_path, f"Stable-diffusion", sd_model_checkpoint)
    sd_save_path = os.path.join(sliders_models_path, "stable-diffusion-v1-5")
    original_config_path = os.path.join(sliders_models_path, "config/v1-inference.yaml")

    os.makedirs(weights_save_path, exist_ok=True)
    os.makedirs(sliders_path, exist_ok=True)
    os.makedirs(low_data_backup_path, exist_ok=True)
    os.makedirs(high_data_backup_path, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(webui_save_path)), exist_ok=True)

    if train_mode_choose == "Train Image Sliders":
        matched_paths_low = []
        matched_paths_high = []

        file_path_mapping_low = {os.path.basename(d['name']): d['name'] for d in low_instance_images}

        for item in high_instance_images:
            file_name = os.path.basename(item['name'])
            if file_name in file_path_mapping_low:
                matched_paths_low.append(file_path_mapping_low[file_name])
                matched_paths_high.append(item['name'])

        for index, user_image in enumerate(matched_paths_low):
            image = Image.open(user_image)
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.save(os.path.join(low_data_backup_path, str(index) + ".jpg"))

        for index, user_image in enumerate(matched_paths_high):
            image = Image.open(user_image)
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.save(os.path.join(high_data_backup_path, str(index) + ".jpg"))
        train_sliders_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_sliders/train_image_sliders.py")
    else:
        train_sliders_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_sliders/train_text_sliders.py")
    print("train_file_path : ", train_sliders_path)

    # outputs/easyphoto-tmp/train_sliders_log.txt, use to cache log and flush to UI
    print("cache_log_file_path:", cache_log_file_path)
    if not os.path.exists(os.path.dirname(cache_log_file_path)):
        os.makedirs(os.path.dirname(cache_log_file_path), exist_ok=True)

    env = None

    if platform.system() == "Windows":
        pwd = os.getcwd()
        dataloader_num_workers = 0  # for solve multi process bug
        command = [
            f"{python_executable_path}",
            "-m",
            "accelerate.commands.launch",
            "--mixed_precision=bf16",
            "--main_process_port=3456",
            f"{train_sliders_path}",
            f"--pretrained_model_path={os.path.relpath(webui_load_path, pwd)}",
            f"--original_config_file={original_config_path}",
            f"--clip_tokenizer_path={sd_save_path}",
            f"--resolution={resolution}",
            f"--batch_size={train_batch_size}",
            f"--max_train_steps={max_train_steps}",
            f"--checkpointing_steps={checkpointing_steps}",
            f"--learning_rate={learning_rate}",
            f"--target_prompt={target_prompt}",
            f"--positive_prompt={positive_prompt}",
            f"--neutral_prompt={neutral_prompt}",
            f"--unconditional_prompt={unconditional_prompt}",
            "--seed=42",
            f"--rank={rank}",
            f"--network_alpha={network_alpha}",
            f"--output_dir={os.path.relpath(weights_save_path, pwd)}",
            "--enable_xformers_memory_efficient_attention",
            "--mixed_precision=bf16",
            f"--cache_log_file={cache_log_file_path}",
        ]
        if train_mode_choose == "Train Image Sliders":
            command += [f'--train_data_dir={sliders_path}']
        try:
            subprocess.run(command, env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing the command: {e}")

    else:
        command = [
            f"{python_executable_path}",
            "-m",
            "accelerate.commands.launch",
            "--mixed_precision=bf16",
            "--main_process_port=3456",
            f"{train_sliders_path}",
            f"--pretrained_model_path={webui_load_path}",
            f"--original_config_file={original_config_path}",
            f"--clip_tokenizer_path={sd_save_path}",
            f"--resolution={resolution}",
            f"--batch_size={train_batch_size}",
            f"--max_train_steps={max_train_steps}",
            f"--checkpointing_steps={checkpointing_steps}",
            f"--learning_rate={learning_rate}",
            f"--target_prompt={target_prompt}",
            f"--positive_prompt={positive_prompt}",
            f"--neutral_prompt={neutral_prompt}",
            f"--unconditional_prompt={unconditional_prompt}",
            "--seed=42",
            f"--rank={rank}",
            f"--network_alpha={network_alpha}",
            f"--output_dir={weights_save_path}",
            "--enable_xformers_memory_efficient_attention",
            "--mixed_precision=bf16",
            f"--cache_log_file={cache_log_file_path}",
        ]
        if train_mode_choose == "Train Image Sliders":
            command += [f'--train_data_dir={sliders_path}']
        try:
            subprocess.run(command, env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing the command: {e}")

    best_weight_path = os.path.join(weights_save_path, f"checkpoint-{max_train_steps}.safetensors")

    if not os.path.exists(best_weight_path):
        return "Failed to obtain Lora after training, please check the training process."

    copyfile(best_weight_path, webui_save_path)
    return "The training has been completed."