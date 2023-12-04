import hashlib
import logging
import os
import requests

from modelscope.utils.logger import get_logger as ms_get_logger
from tqdm import tqdm

from scripts.sliders_config import data_path, models_path

# Ms logger set
ms_logger = ms_get_logger()
ms_logger.setLevel(logging.ERROR)

# ep logger set
sliders_logger_name = __name__.split(".")[0]
sliders_logger = logging.getLogger(sliders_logger_name)
sliders_logger.propagate = False

for handler in sliders_logger.root.handlers:
    if type(handler) is logging.StreamHandler:
        handler.setLevel(logging.ERROR)

stream_handler = logging.StreamHandler()
handlers = [stream_handler]

for handler in handlers:
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))
    handler.setLevel("INFO")
    sliders_logger.addHandler(handler)

sliders_logger.setLevel("INFO")

download_urls = {
    # The models are from civitai/6424, we saved them to oss for your convenience in downloading the models.
    "base": [
        # base model
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/ChilloutMix-ni-fp16.safetensors"
    ]
}
save_filenames = {
    # The models are from civitai/6424, we saved them to oss for your convenience in downloading the models.
    "base": [
        # base model
        os.path.join(models_path, f"Stable-diffusion/Chilloutmix-Ni-pruned-fp16-fix.safetensors")
    ]
}


def check_sliders_valid(lora_path):
    if not lora_path.startswith("sliders_"):
        return False
    return True


def urldownload_progressbar(url, file_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(1024):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))

    progress_bar.close()


def check_files_exists_and_download(check_hash, download_mode="base"):
    urls, filenames = download_urls[download_mode], save_filenames[download_mode]

    # This print will introduce some misundertand
    # print("Start Downloading weights")
    for url, filename in zip(urls, filenames):
        if type(filename) is str:
            filename = [filename]

        exist_flag = False
        for _filename in filename:
            if not check_hash:
                if os.path.exists(_filename):
                    exist_flag = True
                    break
            else:
                if os.path.exists(_filename) and compare_hash_link_file(url, _filename):
                    exist_flag = True
                    break
        if exist_flag:
            continue

        sliders_logger.info(f"Start Downloading: {url}")
        os.makedirs(os.path.dirname(filename[0]), exist_ok=True)
        urldownload_progressbar(url, filename[0])


# Calculate the hash value of the download link and downloaded_file by sha256
def compare_hash_link_file(url, file_path):
    r = requests.head(url)
    total_size = int(r.headers["Content-Length"])

    res = requests.get(url, stream=True)
    remote_head_hash = hashlib.sha256(res.raw.read(1000)).hexdigest()
    res.close()

    end_pos = total_size - 1000
    headers = {"Range": f"bytes={end_pos}-{total_size-1}"}
    res = requests.get(url, headers=headers, stream=True)
    remote_end_hash = hashlib.sha256(res.content).hexdigest()
    res.close()

    with open(file_path, "rb") as f:
        local_head_data = f.read(1000)
        local_head_hash = hashlib.sha256(local_head_data).hexdigest()

        f.seek(end_pos)
        local_end_data = f.read(1000)
        local_end_hash = hashlib.sha256(local_end_data).hexdigest()

    if remote_head_hash == local_head_hash and remote_end_hash == local_end_hash:
        sliders_logger.info(f"{file_path} : Hash match")
        return True

    else:
        sliders_logger.info(f" {file_path} : Hash mismatch")
        return False

