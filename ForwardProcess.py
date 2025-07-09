import torch
from torch.export import dims
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize,RandomHorizontalFlip
import numpy as np

# 定义一个线性beta调度函数，生成一个线性增长的beta值序列。
def linear_beta_schedule(timesteps):
    # 设置beta值的起始和结束值。
    beta_start = 0.0001
    beta_end = 0.02
    # 返回一个从起始值到结束值线性增长的序列。
    return torch.linspace(beta_start, beta_end, timesteps)

# 获取相关参数数值；
def GetElements(timesteps=300):
    betas = linear_beta_schedule(timesteps=timesteps)

    # 通过公式定义\bar{alpha}_t。
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0) # 返回逐步累乘结果；
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # 计算q(x_t | x_{t-1})；
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # 计算 q(x_{t-1} | x_t, x_0)；
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    return [betas,# β参数，控制噪声的加入;
            alphas,
            alphas_cumprod,
            alphas_cumprod_prev,
            sqrt_recip_alphas,  # α的平方根的倒数;
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,  # 1-α的累积乘积的平方根;
            posterior_variance] # 后验方差;

# 定义前向加噪过程，即一步骤到位的加噪过程；
def q_sample(x_start, 
             t, 
             sqrt_alphas_cumprod,
             sqrt_one_minus_alphas_cumprod,
             noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

# 从输入张量a中提取特定的元素，并将它们重塑成与输入张量 x_shape 相关的形状；
def extract(a, t, x_shape):
    batch_size = t.shape[0] # 获取批次大小，即t张量的第一个维度的大小；
    # 使用gather函数根据t张量中的索引来提取a张量中的元素；
    out = a.gather(-1, t.cpu())
    # 重塑提取出的元素，使其形状与x_shape的前n-1个维度相匹配，同时保持批次大小不变；
    reshaped_out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    return reshaped_out.to(t.device)

# 将一张图像转化为一个张量；
def image2tensor(image,image_size=64):
    transform = Compose([
        Resize(image_size),
        CenterCrop(image_size),
        RandomHorizontalFlip(),
        ToTensor(),
        Lambda(lambda t: (t * 2) - 1),
    ])
    return transform(image)

# 将一个张量转化为一张图像；
def tensor2image(tensor):
    reverse_transform = Compose([
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])
    return reverse_transform(tensor.squeeze())

