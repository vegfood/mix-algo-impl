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