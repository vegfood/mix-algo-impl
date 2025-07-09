import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
from einops.layers.torch import Rearrange
from einops import rearrange


# 辅助函数；
def default(val, d):
    if val is not None:
        return val  # 如果val存在，就返回val;

    # 如果val不存在，就检查d是否是一个函数；
    # 如果是函数就调用返回，不是函数就直接返回值；
    return d() if isfunction(d) else d


# 上采样部分；
def Upsample(dim, dim_out=None):
    return nn.ModuleList([
        # 上采样层，缩放因子扩大两倍；
        nn.Upsample(scale_factor=2, mode="nearest"),
        # 卷积；
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)])

# 下采样部分；
def Downsample(dim, dim_out=None):
    return nn.ModuleList([
        # 表示将批次大小为b，通道数为c，高度为h，宽度为w的特征图，
        # 按p1 和p2的值（这里都是2）来重排。
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        # 卷积；
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)])

# 对于时间的位置编码；
class Time_Positional_Encoding(nn.Module):
    def __init__(self,dim):
        super(Time_Positional_Encoding,self).__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device # 同步设备是GPU还是CPU；
        half_dim = self.dim // 2 # 将时间位置编码的维度除上2；

        # 同transformer一样的位置编码的计算公式和方法；
        TPE = math.log(10000) / (half_dim - 1)
        TPE = torch.exp(torch.arange(half_dim, device=device) * -TPE)
        TPE = time[:, None] * TPE[None, :]
        TPE = torch.cat((TPE.sin(), TPE.cos()), dim=-1)
        return TPE
    
# 获得标准化的权重初始化的卷积层；
# 相当于对每个输出通道的初始权重做归一化处理。
class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, x):
        # eps为防止方差为0的“保险”；
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = weight.mean(dim=[1, 2, 3], keepdim=True)
        var = weight.var(dim=[1, 2, 3], keepdim=True, unbiased=False)
        normalized_weight = (weight - mean) / torch.sqrt(var + eps)
        
        # 返回权重初始化后的二维卷积层；
        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

# 一个函数块；
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super(Block,self).__init__()
        # 标准化卷积层；
        self.StdConv2d = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        # 归一化层；
        self.norm = nn.GroupNorm(groups, dim_out)
        # 激活层；
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.StdConv2d(x)
        x = self.norm(x)
        # 将时间作为调整信息嵌入到模块中来；
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x

# 一个残差网络块；
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super(ResnetBlock,self).__init__()
        # 初始化 self.mlp；
        if time_emb_dim is not None:
            # 如果 time_emb_dim 存在，创建一个包含 SiLU 激活和线性变换的序列;
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, dim_out * 2))
        else:
            # 如果 time_emb_dim 不存在，self.mlp 为 None;
            self.mlp = None

        # 两个Block块；
        self.block1 = Block(dim, dim_out, groups=groups)

        self.block2 = Block(dim_out, dim_out, groups=groups)
        # 一个卷积层；
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    # 前向通道；
    def forward(self, x, time_emb=None):
        scale_shift = None
        if (self.mlp is not None) and (time_emb is not None):
            time_emb = self.mlp(time_emb)
            # 重塑成4维，方便进行卷积；
            time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
            # print("time_emb shape:",np.shape(time_emb))
            # 使用chunk方法，将其在channels维度上将其分割为两个维度；
            scale_shift = time_emb.chunk(2, dim=1) 

        h = self.block1(x, scale_shift=scale_shift)
        # print("h in ResBlock1:",np.shape(h))
        h = self.block2(h)
        # print("h in ResBlock2:",np.shape(h))
        return h + self.res_conv(x)

# 添加自注意力模块；
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super(Attention,self).__init__()
        # 缩放因子，用于查询张量的缩放；
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads# 隐藏层的维度；
        # 卷积层，用于生成查询、键和值张量；
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        # 输出用的卷积层；
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape# 获取输入的维度信息；
        # 通过卷积层生成查询、键和值张量，并调整形状；
        qkv = self.to_qkv(x).view(b, self.heads, -1, 3, h * w)
        q, k, v = qkv.unbind(dim=3)
        q = q * self.scale

        sim = torch.matmul(q.transpose(-2, -1), k)# 计算查询和键张量之间的相似度；
        sim = sim - torch.max(sim, dim=-1, keepdim=True)[0]
        attn = torch.softmax(sim, dim=-1)

        # 根据注意力权重和值张量计算输出；
        out = torch.matmul(attn, v.transpose(-2, -1))
        out = out.transpose(-2, -1).contiguous().view(b, -1, h, w)
        return self.to_out(out)
    
    # 添加线性注意力层；
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention,self).__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads # 计算隐藏层维度;
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        # 输出层卷积和归一化;
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1), 
            nn.GroupNorm(1, dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

# Group normalization；
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# 残差结构；
class Residual(nn.Module):
    def __init__(self,fn,dropout=0.1):
        super(Residual,self).__init__()
        self.fn = fn
        self.dropout = nn.Dropout(dropout)

    # *args用来传递任意数量的值，**kwargs用来传递任意数量的键值；
    def forward(self,x,*args, **kwargs):
        return x + self.dropout(self.fn(x,*args, **kwargs))

class Unet(nn.Module):
    def __init__(
        self,
        dim, # 特征的维度；
        init_dim=None, # 初始化的特征维度；
        out_dim=None, # 输出结果的特征维度；
        dim_mults=(1, 2, 4, 8), # 每一个下采样步骤中的特征维度的倍数；
        channels=3, # 输入图像的通道数，默认为3（RGB）；
        self_condition=False,# 是否自我条件化，用于控制输入通道数;
        resnet_block_groups=4, # ResnetBlock的组数；
    ):
        super(Unet,self).__init__()
        self.channels = channels
        self.self_condition = self_condition
        time_dim = dim * 4 # 时间嵌入的维度；
        input_channels = channels * (2 if self_condition else 1)# 根据条件化标志计算输入通道数;
        init_dim = default(init_dim, dim)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]# 计算每个下采样步骤的特征维度;
        in_out = list(zip(dims[:-1], dims[1:]))# 创建输入输出维度对（每一个采样层的dim_in 与 dim_out）;

        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)# 初始卷积层;
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)# 创建ResNet块,即有这么多的块组装的网络层；
        self.time_mlp = nn.Sequential( # 时间嵌入层；
            Time_Positional_Encoding(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.downs = nn.ModuleList([]) # 下采样部分；
        self.ups = nn.ModuleList([]) # 上采样部分；
        num_resolutions = len(in_out) # 上下采样层数；

        ########################### 开始构建下采样层：###########################
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1) # 判断是否是最后一个下采样层；

            self.downs.append(
            nn.ModuleList(
                [   # 如果不是最后一层的话；
                block_klass(dim_in, dim_in, time_emb_dim=time_dim), # ResNet块；
                block_klass(dim_in, dim_in, time_emb_dim=time_dim), # ResNet块；
                Residual(PreNorm(dim_in, LinearAttention(dim_in))), # 带有线性注意力机制的残差块；
                Downsample(dim_in, dim_out) # 下采样一层；
                if not is_last
                # 如果是最后一层的话，一个简单的卷积层就可以了；
                else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                ]
            )
            )
        
        ############################# 开始构建中间层：###########################
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim) # ResNet块；
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))  # 带有注意力机制的残差块；
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim) # ResNet块；

        ########################### 开始构建上采样层：###########################
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
            nn.ModuleList(
                [
                # 如果不是最后一层的话；
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),# ResNet块；
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),# ResNet块；
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),# 带有线性注意力机制的残差块；
                Upsample(dim_out, dim_in) # 上采样一层；
                if not is_last
                # 如果是最后一层的话，一个简单的卷积层就可以了；
                else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                ]
            )
            )
        
        self.out_dim = default(out_dim, channels)# 获得最终的输出维度；
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)# 通过一个ResNet块；
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1) # 使用1x1的卷积核获得最终输出；

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition: # 自我条件化的话，将对应张量合并。
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
            # x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat([x_self_cond, x], dim=1)

        x = self.init_conv(x) # 通过初始卷积层处理输入x；
        r = x.clone()
        # print("time shape in init:",np.shape(time))
        # print("x shape in init:",np.shape(x))
        t = self.time_mlp(time)  # 通过时间MLP处理时间嵌入；
        # print("t shape in init:",np.shape(t))
        h = [] # 初始化一个列表来存储中间特征；

        ###################### 开始下采样过程；######################
        for i in range(len(self.downs)):
            block1, block2, attn, downsample = self.downs[i]
            x = block1(x, t) # 应用第一个ResNet块；
            # print("x shape in block1:",np.shape(x))
            h.append(x) # 将特征添加到h列表；
            x = block2(x, t)  # 应用第二个ResNet块；
            # print("x shape in block2:",np.shape(x))
            x = attn(x)  # 应用注意力机制；
            # print("x shape in down attn:",np.shape(x))
            h.append(x)  # 再次将特征添加到h列表；
            # 手动调用 ModuleList 中的每一层
            if isinstance(downsample, nn.ModuleList):
                for layer in downsample:
                    x = layer(x)  # 先 Rearrange，再 Conv2d
            else:
                x = downsample(x)  # 普通卷积层
            # x = downsample(x) # 下采样；
            # print("x shape in down layer:",np.shape(x))

        ###################### 开始中间层处理；######################
        # print("x shape in mid layer:",np.shape(x))
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        ###################### 开始上采样过程；######################
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1) # 将特征与h列表中最后一个特征合并；
            x = block1(x, t)  # 应用第一个ResNet块；

            x = torch.cat((x, h.pop()), dim=1)  # 再次将特征与h列表中最后一个特征合并；
            x = block2(x, t)  # 应用第二个ResNet块；
            x = attn(x) # 应用注意力机制；

            # x = upsample(x) # 上采样；
            # 手动调用 ModuleList 中的每一层
            if isinstance(upsample, nn.ModuleList):
                for layer in upsample:
                    x = layer(x)  # 先 Rearrange，再 Conv2d
            else:
                x = upsample(x)  # 普通卷积层

        x = torch.cat((x, r), dim=1) # 将特征与初始复制的特征r合并；
        x = self.final_res_block(x, t)  # 应用最终的ResNet块；
        x = self.final_conv(x) # 通过最终的卷积层处理并返回结果。
        return x
