import glob
import os

import gradio as gr
import modules.generation_parameters_copypaste as parameters_copypaste
from modules import script_callbacks, shared
from modules.ui_components import ToolButton as ToolButton_webui

from scripts.sliders_config import (
    cache_log_file_path,
    models_path,
)
from scripts.sliders_train import sliders_train_forward
from scripts.sdwebui import get_checkpoint_type, get_scene_prompt

gradio_compat = True

try:
    from distutils.version import LooseVersion

    from importlib_metadata import version

    if LooseVersion(version("gradio")) < LooseVersion("3.10"):
        gradio_compat = False
except ImportError:
    pass


def get_external_ckpts():
    external_checkpoints = []
    external_ckpt_dir = shared.cmd_opts.ckpt_dir if shared.cmd_opts.ckpt_dir else []
    if len(external_ckpt_dir) > 0:
        for _checkpoint in os.listdir(external_ckpt_dir):
            if _checkpoint.endswith(("pth", "safetensors", "ckpt")):
                external_checkpoints.append(_checkpoint)
    return external_checkpoints


external_checkpoints = get_external_ckpts()


def checkpoint_refresh_function():
    checkpoints = []
    models_dir = os.path.join(models_path, "Stable-diffusion")

    for root, dirs, files in os.walk(models_dir):
        for _checkpoint in files:
            if _checkpoint.endswith(("pth", "safetensors", "ckpt")):
                rel_path = os.path.relpath(os.path.join(root, _checkpoint), models_dir)
                checkpoints.append(rel_path)

    return gr.update(choices=list(set(["Chilloutmix-Ni-pruned-fp16-fix.safetensors"] + checkpoints + external_checkpoints)))


checkpoints = []
models_dir = os.path.join(models_path, "Stable-diffusion")

for root, dirs, files in os.walk(models_dir):
    for _checkpoint in files:
        if _checkpoint.endswith(("pth", "safetensors", "ckpt")):
            rel_path = os.path.relpath(os.path.join(root, _checkpoint), models_dir)
            checkpoints.append(rel_path)


def upload_file(files, current_files):
    file_paths = [file_d["name"] for file_d in current_files] + [file.name for file in files]
    return file_paths


def refresh_display():
    if not os.path.exists(os.path.dirname(cache_log_file_path)):
        os.makedirs(os.path.dirname(cache_log_file_path), exist_ok=True)
    lines_limit = 3
    try:
        with open(cache_log_file_path, "r", newline="") as f:
            lines = []
            for s in f.readlines():
                line = s.replace("\x00", "")
                if line.strip() == "" or line.strip() == "\r":
                    continue
                lines.append(line)

            total_lines = len(lines)
            if total_lines <= lines_limit:
                chatbot = [(None, "".join(lines))]
            else:
                chatbot = [(None, "".join(lines[total_lines - lines_limit :]))]
            return chatbot
    except Exception:
        with open(cache_log_file_path, "w") as f:
            pass
        return None


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", elem_classes=kwargs.pop("elem_classes", []) + ["cnet-toolbutton"], **kwargs)

    def get_block_name(self):
        return "button"


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as sliders_tabs:
        with gr.TabItem("Train"):
            dummy_component = gr.Label(visible=False)
            with gr.Blocks():
                with gr.Row():
                    uuid = gr.Text(label="User_ID", value="", visible=False)

                    with gr.Column():
                        title_of_left_part = gr.Markdown("Training Text Sliders")

                        with gr.Row(visible=False) as instance_images:
                            low_instance_images = gr.Gallery(label="Low instance Images").style(columns=[4], rows=[2], object_fit="contain", height="auto")
                            high_instance_images = gr.Gallery(label="High instance Images").style(columns=[4], rows=[2], object_fit="contain", height="auto")

                        with gr.Row(visible=False) as buttons:
                            low_upload_button = gr.UploadButton("Upload Low Photos", file_types=["image"], file_count="multiple")
                            high_upload_button = gr.UploadButton("Upload High Photos", file_types=["image"], file_count="multiple")
                            clear_button = gr.Button("Clear Photos")

                            low_upload_button.upload(upload_file, inputs=[low_upload_button, low_instance_images], outputs=low_instance_images, queue=False)
                            high_upload_button.upload(upload_file, inputs=[high_upload_button, high_instance_images], outputs=high_instance_images, queue=False)
                            clear_button.click(fn=lambda: [[], []], inputs=None, outputs=[low_instance_images, high_instance_images])

                        with gr.Row():
                            target_prompt = gr.Textbox(
                                label="Target prompt. (Such as: person)",
                                value="person",
                            )

                            positive_prompt = gr.Textbox(
                                label="Positive prompt. (Such as: old person)",
                                value="old person",
                            )

                        with gr.Row():
                            neutral_prompt = gr.Textbox(
                                label="Neutral prompt. (Such as: person)",
                                value="person",
                            )

                            unconditional_prompt = gr.Textbox(
                                label="Unconditional prompt. (Such as: young person)",
                                value="young person",
                            )

                        text_sliders_note = gr.Markdown(
                            """
                            Training steps:
                            1. Please upload 15-20 photos with human, and please don't make the proportion of your face too small.
                            2. Click on the Start Training button below to start the training process, approximately 25 minutes.
                            3. Switch to Inference and generate photos based on the scene lora.
                            4. If you encounter lag when uploading, please modify the size of the uploaded pictures and try to limit it to 1.5MB.
                            """,
                            visible=False,
                        )

                    with gr.Column():
                        title_of_right_part = gr.Markdown("Params Setting")

                        with gr.Accordion("Advanced Options", open=True):
                            with gr.Row():
                                train_mode_choose = gr.Dropdown(
                                    value="Train Text Sliders",
                                    elem_id="radio",
                                    scale=1,
                                    min_width=20,
                                    choices=["Train Text Sliders", "Train Image Sliders"],
                                    label="The Type of Lora",
                                    visible=True,
                                )

                                sd_model_checkpoint = gr.Dropdown(
                                    value="Chilloutmix-Ni-pruned-fp16-fix.safetensors",
                                    scale=3,
                                    choices=list(set(["Chilloutmix-Ni-pruned-fp16-fix.safetensors"] + checkpoints + external_checkpoints)),
                                    label="The base checkpoint you use.",
                                    visible=True,
                                )

                                checkpoint_refresh = ToolButton(value="\U0001f504")
                                checkpoint_refresh.click(fn=checkpoint_refresh_function, inputs=[], outputs=[sd_model_checkpoint])

                            with gr.Row():
                                resolution = gr.Textbox(label="resolution", value=512, interactive=True)
                                checkpointing_steps = gr.Textbox(label="checkpointing steps", value=500, interactive=True)
                                max_train_steps = gr.Textbox(label="max train steps", value=1000, interactive=True)
                                train_batch_size = gr.Textbox(label="train batch size", value=1, interactive=True)
                            with gr.Row():
                                learning_rate = gr.Textbox(label="learning rate", value=2e-4, interactive=True)
                                rank = gr.Textbox(label="rank", value=4, interactive=True)
                                network_alpha = gr.Textbox(label="network alpha", value=1, interactive=True)

                        text_sliders_params_note = gr.Markdown(
                            """
                            Parameter parsing:
                            - **max train steps** represents the maximum training step.
                            """
                        )
                        def update_train_mode(train_mode_choose):
                            if train_mode_choose == "Train Text Sliders":
                                return [
                                    gr.update(value="Training Text Sliders"),
                                    gr.update(value=512),
                                    gr.update(visible=False),
                                    gr.update(visible=False)
                                ]
                            else:
                                return [
                                    gr.update(value="Training Image Sliders"),
                                    gr.update(value=256),
                                    gr.update(visible=True),
                                    gr.update(visible=True)
                                ]

                        train_mode_choose.change(
                            update_train_mode,
                            inputs=train_mode_choose,
                            outputs=[
                                title_of_left_part,
                                resolution,
                                instance_images,
                                buttons,
                            ],
                        )


                with gr.Row():
                    with gr.Column(width=3):
                        run_button = gr.Button("Start Training", variant="primary")
                    with gr.Column(width=1):
                        refresh_button = gr.Button("Refresh Log", variant="primary")

                gr.Markdown(
                    """
                    We need to train first to predict, please wait for the training to complete, thank you for your patience.
                    """
                )
                output_message = gr.Markdown()

                with gr.Box():
                    logs_out = gr.Chatbot(label="Training Logs", height=200)
                    block = gr.Blocks()
                    with block:
                        block.load(refresh_display, None, logs_out, every=3)

                    refresh_button.click(fn=refresh_display, inputs=[], outputs=[logs_out])

                run_button.click(
                    fn=sliders_train_forward,
                    _js="ask_for_style_name",
                    inputs=[
                        sd_model_checkpoint,
                        dummy_component,
                        low_instance_images,
                        high_instance_images,
                        uuid,
                        train_mode_choose,
                        target_prompt, positive_prompt, neutral_prompt, unconditional_prompt,
                        resolution,
                        checkpointing_steps,
                        max_train_steps,
                        train_batch_size,
                        learning_rate,
                        rank,
                        network_alpha
                    ],
                    outputs=[output_message],
                )

    return [(sliders_tabs, "Sliders", f"Sliders_tabs")]


# Configuration items for registration settings page
def on_ui_settings():
    section = ("Sliders", "Sliders")

script_callbacks.on_ui_settings(on_ui_settings)  # 注册进设置页
script_callbacks.on_ui_tabs(on_ui_tabs)
