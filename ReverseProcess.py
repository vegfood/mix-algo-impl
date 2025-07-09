from tqdm import tqdm

from ForwardProcess import *


@torch.no_grad()
def p_sample(model,
             x,  # 当前的样本;
             t,  # 当前的时间步;
             betas,
             sqrt_recip_alphas,
             sqrt_one_minus_alphas_cumprod,
             posterior_variance,
             t_index,
             timesteps=300):
    betas_t = extract(betas, t, x.shape)  # 提取当前时间步的β;
    sqrt_one_minus_alphas_cumprod_t = extract(  # 提取当前时间步的1-α的累积乘积的平方根;
        sqrt_one_minus_alphas_cumprod, t, x.shape)
    # 提取当前时间步的α的平方根的倒数;
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    model_mean = sqrt_recip_alphas_t * (  # 计算模型的均值;
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:  # 如果是第一个时间步;
        return model_mean  # 直接返回模型的均值;
    else:  # 如果不是第一个时间步;
        # 提取当前时间步的后验方差;
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)  # 生成与x形状相同的随机噪声;

        # 返回带有噪声的模型均值;
        return model_mean + torch.sqrt(posterior_variance_t) * noise


# 执行整个模型采样过程；
@torch.no_grad()
def p_sample_loop(model,
                  shape,
                  betas,
                  sqrt_recip_alphas,
                  sqrt_one_minus_alphas_cumprod,
                  posterior_variance,
                  timesteps=300):
    device = next(model.parameters()).device

    b = shape[0]
    img = torch.randn(shape, device=device)
    imgs = []

    # 添加加载条加载进度；
    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model,
                       img,
                       torch.full((b,), i, device=device, dtype=torch.long),
                       betas,
                       sqrt_recip_alphas,
                       sqrt_one_minus_alphas_cumprod,
                       posterior_variance,
                       i)
        imgs.append(img.cpu().numpy())
    return imgs


# 执行采样；
@torch.no_grad()
def sample(model,
           image_size,
           betas,
           sqrt_recip_alphas,
           sqrt_one_minus_alphas_cumprod,
           posterior_variance,
           batch_size=16,
           channels=3):
    return p_sample_loop(model,
                         (batch_size, channels, image_size, image_size),
                         betas,
                         sqrt_recip_alphas,
                         sqrt_one_minus_alphas_cumprod,
                         posterior_variance)
