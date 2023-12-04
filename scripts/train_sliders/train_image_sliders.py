import argparse
import copy
import logging
import gc
import os
import time
import random
import numpy as np
from PIL import Image
from packaging import version
from pathlib import Path
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import randn_tensor

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers import DDIMScheduler, SchedulerMixin, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm
from transformers import CLIPTokenizer

from lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV
from prompt_util import PromptEmbedsCache, PromptEmbedsPair, PromptSettings


device = "cuda"
logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train text sliders.")
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        required=True,
        help="Path to pretrained model checkpoint.",
    )
    parser.add_argument(
        "--original_config_file",
        type=str,
        required=True,
        help="Path to .yaml config file corresponding to the original architecture.",
    )
    parser.add_argument(
        "--clip_tokenizer_path",
        type=str,
        required=True,
        help="Path to CLIP tokenizer.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--network_alpha",
        type=int,
        default=1,
        help="The weight of the LoRA.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1000,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=100,
        help="Save a checkpoint of the training state every X updates.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    parser.add_argument(
        "--target_prompt",
        type=str,
        default=None,
        help="Word for enhancing/erasing the positive concept.",
    )
    parser.add_argument(
        "--positive_prompt",
        type=str,
        required=True,
        help="Concept to enhance",
    )
    parser.add_argument(
        "--unconditional_prompt",
        type=str,
        default=None,
        help="Word to take the difference from the positive concept.",
    )
    parser.add_argument(
        "--neutral_prompt",
        type=str,
        default=None,
        help="Starting point for conditioning the target.",
    )
    parser.add_argument(
        "--action",
        type=str,
        default="enhance",
        choices=["enhance", "erase"],
        help="Enhance or erase the concept.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=int,
        default=4,
        help="guidance_scale",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--folders",
        type=str,
        required=False,
        default = 'low, high',
        help="folders with different attribute-scaled images",
    )
    parser.add_argument(
        "--scales",
        type=str,
        required=False,
        default = '-1, 1',
        help="scales for different attribute-scaled images",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch_size.",
    )
    parser.add_argument(
        "--attributes",
        type=str,
        default=None,
        help="Attritbutes to disentangle (comma seperated string)",
    )
    parser.add_argument(
        "--max_denoising_steps",
        type=int,
        default=50,
        help="Max denoising steps.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_dir",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_log_file",
        type=str,
        default="train_kohya_log.txt",
        help="The output log file path. Use the same log file as train_lora.py",
    )

    args = parser.parse_args()
    return args


def encode_prompts(tokenizer: CLIPTokenizer, text_encoder: CLIPTokenizer, prompts: list[str]):
    text_tokens = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    text_embeddings = text_encoder(text_tokens.to(text_encoder.device))[0]

    return text_embeddings


def predict_noise(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    timestep: int,
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,  # unconditional + conditional
    guidance_scale=7.5,
) -> torch.FloatTensor:
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

    # predict the noise residual
    noise_pred = unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=text_embeddings,
    ).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    guided_target = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    return guided_target


@torch.no_grad()
def get_noisy_image(
    img,
    vae,
    generator,
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    total_timesteps: int = 1000,
    start_timesteps=0,
    
    **kwargs,
):
    # latents_steps = []
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    image = img
    im_orig = image
    device = vae.device
    image = image_processor.preprocess(image).to(device)

    init_latents = vae.encode(image).latent_dist.sample(None)
    init_latents = vae.config.scaling_factor * init_latents

    init_latents = torch.cat([init_latents], dim=0)

    shape = init_latents.shape

    noise = randn_tensor(shape, generator=generator, device=device)

    time_ = total_timesteps
    timestep = scheduler.timesteps[time_:time_+1]
    # get latents
    init_latents = scheduler.add_noise(init_latents, noise, timestep)
    
    return init_latents, noise


def main(args):
    args.output_dir = Path(args.output_dir)

    folders = args.folders.split(',')
    folders = [f.strip() for f in folders]

    scales = args.scales.split(',')
    scales = [f.strip() for f in scales]
    scales = [int(s) for s in scales]
    if len(scales) != len(folders):
        raise Exception('the number of folders need to match the number of scales')

    scales = np.array(scales)
    folders = np.array(folders)
    scales_unique = list(scales)

    # accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    # kwargs =DistributedDataParallelKwargs (find_unused_parameters=True)
    # accelerator = Accelerator(
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,
    #     mixed_precision=args.mixed_precision,
    #     log_with=args.report_to,
    #     project_config=accelerator_project_config,
    #     kwargs_handlers=[kwargs],
    # )
    # # Make one log on every process with the configuration for debugging.
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     level=logging.INFO,
    # )
    # logger.info(accelerator.state, main_process_only=False)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load prompts.yaml from argparse. Currently, only one prompt setting is supported.
    raw_prompts = []
    item = {"guidance_scale": args.guidance_scale, "resolution": args.resolution, "batch_size": args.batch_size}
    item["target"] = "" if args.target_prompt is None else args.target_prompt
    item["positive"] = args.positive_prompt
    if args.target_prompt is not None:
        item["positive"] = args.target_prompt + ", " + args.positive_prompt
    item["unconditional"] = item["target"]
    if args.unconditional_prompt is not None:
        item["unconditional"] = args.target_prompt + ", " + args.unconditional_prompt
    item["neutral"] = item["target"] if args.neutral_prompt is None else args.neutral_prompt
    item["action"] = args.action
    raw_prompts.append(copy.deepcopy(item))

    # Multiple types of attributes.
    # attributes = [[]]  "female, male; age1, age2; race1, race2"
    # if args.attributes is not None:
    #     args.attributes = args.attributes.replace(" ", "")
    #     attributes = [item.split(",") for item in args.attributes.split(";") if item]
    # combinations = list(itertools.product(*attributes))
    attributes = []
    if args.attributes is not None:
        attributes = args.attributes.replace(" ", "").split(",")

    # Constrain prompts with attributes to perform the disentanglement.
    prompts = []
    if len(attributes) != 0:
        for i in range(len(raw_prompts)):
            for attr in attributes:
                item = copy.deepcopy(raw_prompts[i])
                item["target"] = attr + " " + item["target"]
                item["positive"] = attr + " " + item["positive"]
                item["neutral"] = attr + " " + item["neutral"]
                item["unconditional"] = attr + " " + item["unconditional"]
                prompts.append(item)
    else:
        prompts = raw_prompts

    prompts = [PromptSettings(**prompt) for prompt in prompts]
    print(prompts)

    # Load scheduler, tokenizer and models.
    tokenizer = CLIPTokenizer.from_pretrained(args.clip_tokenizer_path, subfolder="tokenizer")
    pipeline = download_from_original_stable_diffusion_ckpt(
        args.pretrained_model_path,
        original_config_file=args.original_config_file,
        pipeline_class=StableDiffusionPipeline,
        model_type=None,
        stable_unclip=None,
        controlnet=False,
        from_safetensors=True,
        extract_ema=False,
        image_size=None,
        scheduler_type="pndm",
        num_in_channels=None,
        upcast_attention=None,
        load_safety_checker=False,
        prediction_type=None,
        text_encoder=None,
        tokenizer=tokenizer,
    )
    unet, text_encoder, vae = pipeline.unet, pipeline.text_encoder, pipeline.vae
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        prediction_type="epsilon",
    )
    # del pipeline.vae

    # We only train the additional adapter LoRA layers
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    
    text_encoder.eval()
    unet.eval()

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Create LoRA with type `c3lier`.
    network_type = "c3lier"
    modules = DEFAULT_TARGET_REPLACE
    if network_type == "c3lier":
        modules += UNET_TARGET_REPLACE_MODULE_CONV
    network = LoRANetwork(
        unet,
        rank=args.rank,
        multiplier=1.0,
        alpha=args.network_alpha,
        train_method="noxattn",
    ).to(device, dtype=weight_dtype)

    optimizer = torch.optim.AdamW(network.prepare_optimizer_params(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1)
    criteria = torch.nn.MSELoss()

    cache = PromptEmbedsCache()
    prompt_pairs: list[PromptEmbedsPair] = []

    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py.
    # Encode target, positive, neutral and unconditional prompt.
    with torch.no_grad():
        for settings in prompts:
            for prompt in [
                settings.target,
                settings.positive,
                settings.neutral,
                settings.unconditional,
            ]:
                if isinstance(prompt, list):
                    if prompt == settings.positive:
                        key_setting = "positive"
                    else:
                        key_setting = "attributes"
                    if len(prompt) == 0:
                        cache[key_setting] = []
                    else:
                        if cache[key_setting] is None:
                            cache[key_setting] = encode_prompts(tokenizer, text_encoder, prompt)
                else:
                    if cache[prompt] == None:
                        cache[prompt] = encode_prompts(tokenizer, text_encoder, [prompt])

            prompt_pairs.append(
                PromptEmbedsPair(
                    criteria,
                    cache[settings.target],
                    cache[settings.positive],
                    cache[settings.unconditional],
                    cache[settings.neutral],
                    settings,
                )
            )

    del tokenizer
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    # check log path
    output_log = open(args.cache_log_file, 'w')

    pbar = tqdm(range(args.max_train_steps))
    for i in pbar:
        with torch.no_grad():
            optimizer.zero_grad()
            
            # Prepare timesteps.
            noise_scheduler.set_timesteps(args.max_denoising_steps, device=device)
            timesteps_to = torch.randint(1, args.max_denoising_steps - 1, (1,)).item()

            prompt_pair: PromptEmbedsPair = prompt_pairs[torch.randint(0, len(prompt_pairs), (1,)).item()]

            height, width = (prompt_pair.resolution, prompt_pair.resolution)
            # if prompt_pair.dynamic_resolution:
            #     height, width = train_util.get_random_resolution_in_bucket(
            #         prompt_pair.resolution
            #     )

            scale_to_look = abs(random.choice(list(scales_unique)))
            folder1 = folders[scales==-scale_to_look][0]
            folder2 = folders[scales==scale_to_look][0]
            
            ims = os.listdir(f'{args.train_data_dir}/{folder1}/')
            ims = [im_ for im_ in ims if '.png' in im_ or '.jpg' in im_ or '.jpeg' in im_ or '.webp' in im_]
            random_sampler = random.randint(0, len(ims)-1)

            img1 = Image.open(f'{args.train_data_dir}/{folder1}/{ims[random_sampler]}').resize((width, height))
            img2 = Image.open(f'{args.train_data_dir}/{folder2}/{ims[random_sampler]}').resize((width, height))
            
            seed = random.randint(0, 2 * 15)
            
            generator = torch.manual_seed(seed)
            denoised_latents_low, low_noise = get_noisy_image(
                img1,
                vae,
                generator,
                unet,
                noise_scheduler,
                start_timesteps=0,
                total_timesteps=timesteps_to)
            denoised_latents_low = denoised_latents_low.to(device, dtype=weight_dtype)
            low_noise = low_noise.to(device, dtype=weight_dtype)
            
            generator = torch.manual_seed(seed)
            denoised_latents_high, high_noise = get_noisy_image(
                img2,
                vae,
                generator,
                unet,
                noise_scheduler,
                start_timesteps=0,
                total_timesteps=timesteps_to)
            denoised_latents_high = denoised_latents_high.to(device, dtype=weight_dtype)
            high_noise = high_noise.to(device, dtype=weight_dtype)
            noise_scheduler.set_timesteps(1000)

            current_timestep = noise_scheduler.timesteps[
                int(timesteps_to * 1000 / args.max_denoising_steps)
            ]

            # with network: の外では空のLoRAのみが有効になる
            high_latents = predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents_high,
                torch.cat((prompt_pair.unconditional, prompt_pair.positive)).repeat_interleave(
                    prompt_pair.batch_size, dim=0
                ),
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)
            # with network: の外では空のLoRAのみが有効になる
            low_latents = predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents_low,
                torch.cat((prompt_pair.unconditional, prompt_pair.unconditional,)).repeat_interleave(
                    prompt_pair.batch_size, dim=0
                ),
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)
            
        network.set_lora_slider(scale=scale_to_look)
        with network:
            target_latents_high = predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents_high,
                torch.cat([prompt_pair.unconditional, prompt_pair.positive]).repeat_interleave(
                    prompt_pair.batch_size, dim=0
                ),
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)
            
            
        high_latents.requires_grad = False
        low_latents.requires_grad = False
        
        loss_high = criteria(target_latents_high, high_noise.cpu().to(torch.float32))
        pbar.set_description(f"Loss*1k: {loss_high.item()*1000:.4f}")
        loss_high.backward()
        
        network.set_lora_slider(scale=-scale_to_look)
        with network:
            target_latents_low = predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents_low,
                torch.cat([prompt_pair.unconditional, prompt_pair.neutral]).repeat_interleave(
                    prompt_pair.batch_size, dim=0
                ),
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)
            
        high_latents.requires_grad = False
        low_latents.requires_grad = False
        
        loss_low = criteria(target_latents_low, low_noise.cpu().to(torch.float32))
        pbar.set_description(f"Loss*1k: {loss_low.item()*1000:.4f}")
        loss_low.backward()
        
        time_info = str(time.asctime(time.localtime(time.time())))
        log_line = f"{time_info}: Step {i}: Loss Low * 1k: {loss_low.item()*1000:.4f}. Loss High * 1k: {loss_high.item()*1000:.4f}. \n"
        
        output_log.write(log_line)
        output_log.flush()

        ## NOTICE NO zero_grad between these steps (accumulating gradients) 
        #following guidelines from Ostris (https://github.com/ostris/ai-toolkit)
        optimizer.step()
        lr_scheduler.step()

        del (
            high_latents,
            low_latents,
            target_latents_low,
            target_latents_high,
        )

        if i % args.checkpointing_steps == 0 and i != 0 and i != args.max_train_steps - 1:
            save_path = args.output_dir / "checkpoint-{}.safetensors".format(i)
            network.save_weights(save_path, dtype=weight_dtype)
            print("Saved state to {}".format(save_path))

    print("Saved state to {}".format(save_path))
    save_path = args.output_dir / "checkpoint-{}.safetensors".format(args.max_train_steps)
    network.save_weights(save_path, dtype=weight_dtype)
    
    with open(args.cache_log_file, "w") as _:
        pass

    del (unet, network)
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    main(args)
