import torch
import Loss
import Model
import ForwardProcess as FP
from torch.optim import Adam
from torchvision.utils import save_image

epochs = 40 #确定训练轮次；
timesteps = 600 #确定最大时间步；

# 获得超参数；
elements = FP.GetElements(timesteps=timesteps)
# 其中不同的参数的含义；
# [betas,# β参数，控制噪声的加入;                            0
#  alphas,                                                    1
#  alphas_cumprod,                                            2
#  alphas_cumprod_prev,                                       3
#  sqrt_recip_alphas,  # α的平方根的倒数;                    4
#  sqrt_alphas_cumprod,                                       5
#  sqrt_one_minus_alphas_cumprod,  # 1-α的累积乘积的平方根;  6
#  posterior_variance] # 后验方差;                            7

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

# 确定优化器；
optimizer = Adam(model.parameters(), lr=1e-3)

# 确定dataset;
dataset = Loss.makeDataLoader()

# epochs = 40 #确定训练轮次；
# timesteps = 600 #确定最大时间步；

# 定义一个函数num_to_groups，接受两个参数：num（被分组的数）和divisor（每组的大小）;
def num_to_groups(num, divisor):
    groups = num // divisor 
    remainder = num % divisor  # 使用模运算符'%'计算分组后的余数;
    arr = [divisor] * groups  # 创建一个列表，包含'groups'个'divisor'，即完全分组的列表;
    if remainder > 0: 
        arr.append(remainder)  # 将余数作为一个新的组添加到列表中;
    return arr  # 返回包含所有组的列表;

save_and_sample_every = 1000

for epoch in range(epochs):
    for step, batch in enumerate(dataset):
        optimizer.zero_grad()
        batch_size = batch.shape[0]
        batch = batch.to(device)

        # 任取一个作为时间步进行下降；
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()
        # 获取Loss,并计算得到结果；
        loss = Loss.p_losses(model, 
                             batch, 
                             t, 
                             elements[5],
                             elements[6],
                             loss_type="huber")

        if step % 100 == 0:
            print("Loss:", loss.item())

        # 反向传播；
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(),'./model.pt')
