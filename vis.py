import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
import torchvision.transforms as transforms
# from DCNv2_latest.dcn_v2 import DCNv2
from basicsr.archs.arch_util import to_2tuple, trunc_normal_, flow_warp 
from PIL import Image

from basicsr.utils.registry import ARCH_REGISTRY


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale

def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0.0)

def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError

def batch_index_fill(x, x1, x2, idx1, idx2):
    B, N, C = x.size()
    B, N1, C = x1.size()
    B, N2, C = x2.size()

    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1)
    idx1 = idx1 + offset * N
    idx2 = idx2 + offset * N

    x = x.reshape(B*N, C)

    x[idx1.reshape(-1)] = x1.reshape(B*N1, C)
    x[idx2.reshape(-1)] = x2.reshape(B*N2, C)

    x = x.reshape(B, N, C)
    return x


class PredictorLG(nn.Module):
    """ Importance Score Predictor
    """
    def __init__(self, dim, window_size=8, k=4):
        super().__init__()

        self.window_size = window_size

        cdim = dim + k
        embed_dim = window_size**2

        self.in_conv = nn.Sequential(
            nn.Conv2d(cdim, cdim//4, 1),
            LayerNorm(cdim//4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.out_offsets = nn.Sequential(
            nn.Conv2d(cdim//4, cdim//8, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(cdim//8, 2, 1),
        )

        self.out_mask = nn.Sequential(
            nn.Linear(embed_dim, window_size),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(window_size, 2),
            nn.Softmax(dim=-1)
        )

        self.out_CA = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cdim//4, dim, 1),
            nn.Sigmoid(),
        )

        self.out_SA = nn.Sequential(
            nn.Conv2d(cdim//4, 1, 3, 1, 1),
            nn.Sigmoid(),
        )        


    def forward(self, input_x, mask=None, ratio=0.5, train_mode=False):

        x = self.in_conv(input_x)

        offsets = self.out_offsets(x)
        offsets = offsets.tanh().mul(8.0)

        ca = self.out_CA(x)
        sa = self.out_SA(x)
        
        x = torch.mean(x, keepdim=True, dim=1) 

        x = rearrange(x,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        B, N, C = x.size()

        pred_score = self.out_mask(x)
        mask = F.gumbel_softmax(pred_score, hard=True, dim=2)[:, :, 0:1]

        if self.training or train_mode:
            return mask, offsets, ca, sa
        else:
            ratio = 1.0
            score = pred_score[:, : , 0]
            B, N = score.shape
            r = torch.mean(mask,dim=(0,1))
            num_keep_node = min(N,int(N * r * ratio))
            idx = torch.argsort(score, dim=1, descending=True)
            idx1 = idx[:, :num_keep_node]
            idx2 = idx[:, num_keep_node:]
            return [idx1, idx2], offsets, ca, sa

class PostNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.norm(self.fn(x) + x)

class CAMixer(nn.Module):
    def __init__(self, dim, window_size=8, bias=True, is_deformable=True, ratio=0.5):
        super().__init__()    

        self.dim = dim
        self.window_size = window_size
        self.is_deformable = is_deformable
        self.ratio = ratio
        k = 3
        d = 2

        self.project_v = nn.Conv2d(dim, dim, 1, 1, 0, bias = bias)
        self.project_q = nn.Linear(dim, dim, bias = bias)
        self.project_k = nn.Linear(dim, dim, bias = bias)

        # self.sim_attn = nn.Conv2d(dim, dim, kernel_size=3, bias=True, groups=dim, padding=1)
        
        # Conv
        # self.conv_sptial = nn.Conv2d(dim, dim, kernel_size=3, bias=True, groups=dim, padding=1)
        self.conv_sptial = nn.Sequential(
            nn.Conv2d(dim, dim, k, padding=k//2, groups=dim),
            nn.Conv2d(dim, dim, k, stride=1, padding=((k//2)*d), groups=dim, dilation=d))        
        self.project_out = nn.Conv2d(dim, dim, 1, 1, 0, bias = bias)

        self.act = nn.GELU()
        # Predictor
        self.route = PredictorLG(dim,window_size)

    def forward(self,x,condition_global=None, mask=None, train_mode=False):
        N,C,H,W = x.shape

        v = self.project_v(x)

        if self.is_deformable:
            condition_wind = torch.stack(torch.meshgrid(torch.linspace(-1,1,self.window_size),torch.linspace(-1,1,self.window_size)))\
                    .type_as(x).unsqueeze(0).repeat(N, 1, H//self.window_size, W//self.window_size)
            if condition_global is None:
                _condition = torch.cat([v, condition_wind], dim=1)
            else:
                _condition = torch.cat([v, condition_global, condition_wind], dim=1)

        mask, offsets, ca, sa = self.route(_condition,ratio=self.ratio,train_mode=train_mode)

        q = x 
        k = x + flow_warp(x, offsets.permute(0,2,3,1), interp_mode='bilinear', padding_mode='border')
        qk = torch.cat([q,k],dim=1)

        # Attn branch
        vs = v*sa

        v  = rearrange(v,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        vs = rearrange(vs,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        qk = rearrange(qk,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)

        if self.training or train_mode:
            N_ = v.shape[1]
            v1,v2 = v*mask, vs*(1-mask)   
            qk1 = qk*mask 
        else:
            idx1, idx2 = mask
            _, N_ = idx1.shape
            v1,v2 = batch_index_select(v,idx1),batch_index_select(vs,idx2)
            qk1 = batch_index_select(qk,idx1)

        v1 = rearrange(v1,'b n (dh dw c) -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
        qk1 = rearrange(qk1,'b n (dh dw c) -> b (n dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)

        q1,k1 = torch.chunk(qk1,2,dim=2)
        q1 = self.project_q(q1)
        k1 = self.project_k(k1)
        q1 = rearrange(q1,'b (n dh dw) c -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
        k1 = rearrange(k1,'b (n dh dw) c -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
  

        #calculate attention: Softmax(Q@K)@V
        attn = q1 @ k1.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        f_attn = attn@v1

        f_attn = rearrange(f_attn,'(b n) (dh dw) c -> b n (dh dw c)', 
            b=N, n=N_, dh=self.window_size, dw=self.window_size)

        if not (self.training or train_mode):
            attn_out = batch_index_fill(v.clone(), f_attn, v2.clone(), idx1, idx2)
        else:
            attn_out = f_attn + v2

        attn_out = rearrange(
            attn_out, 'b (h w) (dh dw c) -> b (c) (h dh) (w dw)', 
            h=H//self.window_size, w=W//self.window_size, dh=self.window_size, dw=self.window_size
        )
        
        out = attn_out
        out = self.act(self.conv_sptial(out))*ca + out
        out = self.project_out(out)


        if self.training:
            return out, torch.mean(mask,dim=1)
        return out, [mask,offsets]

class GatedFeedForward(nn.Module):
    def __init__(self, dim, mult = 1, bias=False, dropout = 0.):
        super().__init__()

        hidden_features = int(dim*mult)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Block(nn.Module):
    def __init__(self, n_feats, window_size=8, ratio=0.5):
        super(Block,self).__init__()
        
        self.n_feats = n_feats
        self.norm1 = LayerNorm(n_feats)
        self.mixer = CAMixer(n_feats,window_size=window_size,ratio=ratio)
        self.norm2 = LayerNorm(n_feats)
        self.ffn = GatedFeedForward(n_feats)
        
        
    def forward(self,x, global_condition=None):
        if self.training:
            res, decision = self.mixer(x,global_condition)
            x = self.norm1(x+res)
            res = self.ffn(x)
            x = self.norm2(x+res)
            return x, decision
        else:
            res, decision = self.mixer(x,global_condition)
            x = self.norm1(x+res)
            res = self.ffn(x)
            x = self.norm2(x+res)
            return x, decision


class Group(nn.Module):
    def __init__(self, n_feats, n_block, window_size=8, ratio=0.5):
        super(Group, self).__init__()
        
        self.n_feats = n_feats

        self.body = nn.ModuleList([Block(n_feats, window_size=window_size, ratio=ratio) for i in range(n_block)])
        self.body_tail = nn.Conv2d(n_feats, n_feats, 1, 1, 0)
        
    def forward(self,x,condition_global=None):
        decision = []
        shortcut = x.clone()
        if self.training:
            for _, blk in enumerate(self.body):
                x, mask = blk(x,condition_global)
                decision.append(mask)
            x = self.body_tail(x) + shortcut
            return x, decision
        else:
            for _, blk in enumerate(self.body):
                x, mask = blk(x,condition_global)
                decision.append(mask)
            x = self.body_tail(x) + shortcut
            return x, decision 

#@ARCH_REGISTRY.register()
class OSR(nn.Module):
    def __init__(self, n_block=4, n_group=4, n_colors=3, n_feats=60, scale=4):
        super().__init__()

        self.head = nn.Conv2d(n_colors, n_feats, 3, 1, 1)
        self.window_sizes = [16,16]

        ratios = [0.5 for _ in range(n_group)]
        blocks = [4,4,6,6]

        self.global_predictor = nn.Sequential(nn.Conv2d(n_feats, 8, 1, 1, 0, bias=True),
                                        nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                        nn.Conv2d(8, 2, 3, 1, 1, bias=True),
                                        nn.LeakyReLU(negative_slope=0.1, inplace=True))

        self.scale = scale
        # define body module
        self.body = nn.ModuleList([Group(n_feats, n_block=blocks[i], window_size=self.window_sizes[i%2], ratio=ratios[i]) for i in range(n_group)])
        self.body_tail = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        # define tail module
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_colors*(scale**2), 3, 1, 1),
            nn.PixelShuffle(scale)
        )
        
    def forward(self, x):
        decision = []
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        x = self.head(x)
            
        condition_global = self.global_predictor(x)
        shortcut = x.clone()
        for _, blk in enumerate(self.body):
            x, mask = blk(x,condition_global)
            decision.extend(mask)

        x = self.body_tail(x) 
        x = x + shortcut
        x = self.tail(x)

        if self.training:
            return x[:, :, 0:H*self.scale, 0:W*self.scale], torch.mean(torch.cat(decision,dim=1),dim=(0,1))
        else:
            return x[:, :, 0:H*self.scale, 0:W*self.scale], decision

    def check_image_size(self, x):
        _, _, h, w = x.size()
        wsize = self.window_sizes[0]
        for i in range(1, len(self.window_sizes)):
            wsize = wsize*self.window_sizes[i] // math.gcd(wsize, self.window_sizes[i])
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

# def visual(x,mask):
#     _, _, h, w = x.size()
#     wsize = 16
#     i=16
#     mod_pad_h = (wsize - h % wsize) % wsize
#     mod_pad_w = (wsize - w % wsize) % wsize
#     x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

#     res = x
#     mean_y = x*0


#     B,C,H,W = x.shape
#     origin_x = x
#     x = rearrange(x,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=wsize, dw=wsize)
#     x_input = x
#     print(x.shape)
#     p =  len(mask)
#     for i in range(p):
#         # i=18
#         idx1,idx2 = mask[i][0]
#         offset = mask[i][1]
#         x1,x2 = batch_index_select(x_input,idx1), batch_index_select(x_input,idx2)
#         print(x1.shape[1],x_input.shape[1])
#         x2 = x2*0.1 + 0.9
#         x1 = x1*1.0
#         x = batch_index_fill(x_input.clone(), x1, x2, idx1, idx2)
#         x = rearrange(x,'b (h w) (dh dw c) -> b c (h dh) (w dw)', h=H//wsize, w=W//wsize, dh=wsize, dw=wsize, c = C)

#         x_warp = flow_warp(origin_x, offset.permute(0,2,3,1), interp_mode='bilinear', padding_mode='border')
#         # x = torch.max(x,res)

#         img1 = transforms.ToPILImage()(x.squeeze(0))
#         # img1.show()
#         img1.save('temp{}.png'.format(i))

def visual(x, mask):
    _, _, h, w = x.size()
    wsize = 16
    mod_pad_h = (wsize - h % wsize) % wsize
    mod_pad_w = (wsize - w % wsize) % wsize
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

    B, C, H, W = x.shape
    origin_x = x
    x = rearrange(x, 'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=wsize, dw=wsize)
    x_input = x

    # 定义颜色和透明度
    alpha = 0.5  # 透明度控制，值越小越透明
    light_red = torch.tensor([1.0, 0.8, 0.8], device=x.device)    # 浅红色
    light_green = torch.tensor([0.8, 1.0, 0.8], device=x.device) # 浅绿色

    # 创建窗口颜色模板
    window_pixels_red = light_red.repeat(wsize*wsize).view(1, 1, -1)
    window_pixels_green = light_green.repeat(wsize*wsize).view(1, 1, -1)

    for i in range(len(mask)):
        idx1, idx2 = mask[i][0]
        offset = mask[i][1]
        
        # 获取要处理的窗口
        x1 = batch_index_select(x_input, idx1)
        x2 = batch_index_select(x_input, idx2)

        # 对非mask区域应用浅lv色（保持原内容+颜色叠加）
        x1 = x1 * (1 - alpha) + window_pixels_green * alpha
        # 对mask区域应用浅绿色（保持原内容+颜色叠加）
        x2 = x2 * (1 - alpha) + window_pixels_red * alpha

        # 创建带颜色叠加的新特征图
        x_colored = batch_index_fill(x_input.clone(), x1, x2, idx1, idx2)

        # 恢复空间布局
        x_vis = rearrange(x_colored, 'b (h w) (dh dw c) -> b c (h dh) (w dw)', 
                        h=H//wsize, w=W//wsize, dh=wsize, dw=wsize, c=C)

        # 可选的warp操作
        x_warp = flow_warp(origin_x, offset.permute(0,2,3,1), 
                          interp_mode='bilinear', padding_mode='border')

        # 保存结果
        img = torch.clamp(x_vis, 0, 1).squeeze(0)
        img = transforms.ToPILImage()(img)
        img.save(f'temp{i}.png')



if __name__ == '__main__':

    f = OSR(scale=2).cuda()
    f.load_state_dict(torch.load(r'CAMixerSRx2.pth')['params_ema'], strict=True)
    f.eval()

    #img = Image.open(r'10.160.15.241C04130-20220511050846_0001.png').convert('RGB')
    img = Image.open(r'239.197.43.37-6310_20220510070000_0009_1.jpg').convert('RGB')

    
    x = transforms.ToTensor()(img)  
    x = x.unsqueeze(0).cuda()

    p1,mask = f(x)
    img1 = transforms.ToPILImage()(p1.squeeze(0))
    # img1.show()
    img1.save('SR.png')
    visual(x,mask)

        