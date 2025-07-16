import torch
import Model
import ForwardProcess as FP
import ReverseProcess as RP
import numpy as np
from PIL import Image
import os

# 在gpu上使用程序；
device = "cuda" if torch.cuda.is_available() else "cpu"

# 确定模型参数；
image_size = 64
channels = 3
model = Model.Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,)
)
model.to(device)

# 加载模型；
model.load_state_dict(torch.load('model.pt'))
model.eval()

# 去噪步数
timesteps = 300
# 获得公式计算的相关参数；
elements = FP.GetElements(timesteps=timesteps)

samples = RP.sample(model,
                    image_size=image_size,
                    batch_size=32,
                    channels=channels,
                    betas=elements[0],
                    sqrt_recip_alphas=elements[4],
                    sqrt_one_minus_alphas_cumprod=elements[6],
                    posterior_variance=elements[7], timesteps=timesteps)

print("shape of samples:",np.shape(samples))
samples = torch.tensor(samples, dtype=torch.float32)

if not os.path.exists('results'):# 当前目录下没有文件夹就创造一个。
    os.makedirs('results')

# 遍历每个时间步
for i in range(samples.shape[0]):
    # 随机选择一个图像;
    image_index = 25
    image = samples[i, image_index]
    # 将数值缩放到0-255之内;
    img_normalized = ((image - image.min()) * (255 / (image.max() - image.min())))
    # 将numpy数组转换为PIL图像;
    img_normalized = img_normalized.numpy().astype(np.int8)
    img_normalized = np.transpose(img_normalized, (1, 2, 0))
    img_pil = Image.fromarray(img_normalized, 'RGB')
    # 保存图像，图像名称为时间步的名称;
    img_pil.save(f'results/time_step_{i}.png')

print("ending.")