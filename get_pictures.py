from huggingface_hub import hf_hub_download
import numpy as np

file = hf_hub_download(
    repo_id="username/imagenet-64-mirror",  # 需搜索可用镜像
    filename="imagenet64_train.npy",
    repo_type="dataset"
)
train_data = np.load(file)