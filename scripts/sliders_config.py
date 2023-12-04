import os

from modules.paths import data_path, models_path

# save_dirs
data_dir = data_path
models_path = models_path
sliders_models_path = os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models")
sliders_id_outpath_samples = os.path.join(data_dir, "outputs/sliders-id-infos")

cache_log_file_path = os.path.join(data_dir, "outputs/sliders-tmp/train_kohya_log.txt")