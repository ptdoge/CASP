import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import datasets, transforms
from models.ops_dcnv3.modules import DCNv3
from typing import List
from models.mit import MixVisionTransformer

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'
param_init = dict(
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
    in_channels=3,
    embed_dims=32,
    num_stages=4,
    num_layers=[2, 2, 2, 2],
    num_heads=[1, 2, 5, 8],
    patch_sizes=[7, 3, 3, 3],
    sr_ratios=[8, 4, 2, 1],
    out_indices=(0, 1, 2, 3),
    mlp_ratio=4,
    qkv_bias=True,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.1)

class SPAligned(nn.Module):
    def __init__(self, in_ch):
        super().__init__() # 
        
        self.q = nn.Linear(in_ch, in_ch//4, bias=False)
        self.k = nn.Linear(in_ch, in_ch//4, bias=False)

        self.v = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, kernel_size=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
            ) 
        
    def forward(self, x, k=7): # 

        xt1, xt2 = torch.chunk(x, 2, dim=0)
        
        xt1 = self.align(xt2, xt1, k)
        xt2 = self.align(xt1, xt2, k) #

        xt = torch.cat([xt1, xt2], dim=0)

        x = self.v(torch.cat([x, xt], dim=1))

        return x 
      
    def align(self, ta, tb, k):

        b, _, h, w = ta.size()
        spta = ta.reshape(b, -1, 1, h*w).permute(0, 3, 2, 1)
        tb = F.unfold(tb, kernel_size=k, padding=(k-1)//2).reshape(b, -1, k**2, h*w).permute(0, 3, 2, 1)
        sptb = tb
        sptb = torch.cat([spta, sptb], dim=-2)
        spta = self.q(spta)
        sptb = self.k(sptb)
        attn = (spta @ sptb.transpose(-1,-2).contiguous()).softmax(dim=-1)
        tb = (attn[:, :, :, 1:] @ tb).permute(0, 3, 2, 1).reshape(b, -1, h, w)

        return tb

class SAFM(nn.Module):
    def __init__(self, in_channels, guidance_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(SAFM, self).__init__() 

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv1 = nn.Sequential(
            nn.Conv2d(guidance_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True) 

        self.theta = theta
        self.guidance_channels = guidance_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        x_initial = torch.randn(in_channels, 1, kernel_size, kernel_size)
        x_initial[:, :, int(kernel_size / 2), int(kernel_size / 2)] = -1
        self.x_kernel_diff = nn.Parameter(x_initial)
        self.x_kernel_diff[:, :, int(kernel_size / 2), int(kernel_size / 2)].detach()

        guidance_initial = torch.randn(in_channels, 1, kernel_size, kernel_size)
        guidance_initial[:, :, int(kernel_size / 2), int(kernel_size / 2)] = -1 
        self.guidance_kernel_diff = nn.Parameter(guidance_initial)
        self.guidance_kernel_diff[:, :, int(kernel_size / 2), int(kernel_size / 2)].detach() 
#  
    def forward(self, x, guidance):

        in_channels = self.in_channels

        guidance = self.conv1(self.up(guidance))

        x_diff = F.conv2d(input=x, weight=self.x_kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=1, groups=in_channels) # 
        guidance_diff = F.conv2d(input=guidance, weight=self.guidance_kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=1, groups=in_channels)

        out = self.conv(x_diff + guidance_diff) #
        out = self.relu(self.bn(out))

        out = out + x

        return out

class DCAlign(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.dcn =DCNv3(in_ch, group=1)
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x1, x2):

        x2 = F.interpolate(x2, x1.shape[-2:], mode="bilinear", align_corners=False) # 
        x21, x22 = torch.chunk(x2, 2, dim=0)
        x2 = torch.abs(x21-x22).repeat(2,1,1,1)    #
        x = self.dcn(x1.permute(0,2,3,1), x2.permute(0,2,3,1)).permute(0,3,1,2) 
        x = self.relu(self.bn(x)) + x1

        return x

class CASP(nn.Module):
    def __init__(self, in_ch, pretrained=True):
        super().__init__()

        self.mit = MixVisionTransformer(**param_init)

        # Segmentation Head
        decoder_dim = 96
        embed_dims = [32, 64, 160, 256] # mitb0
        
        self.be = be = True
        if be == True: 
            self.bem = nn.ModuleList(
                [
                    SAFM(embed_dims[i], embed_dims[i+1]) for i in range(3)
                ]
            )
        
        self.align = align = True
        if align == True:
            self.spalign = nn.ModuleList(
                [
                    SPAligned(decoder_dim) for i in range(1)
                ]
            )

            # new
            self.alignv2 = alignv2 = True
            if alignv2 == True:
                self.dcnalign = nn.ModuleList(
                    [
                        DCAlign(decoder_dim) for i in range(2)
                    ]
                )

        self.to_dr = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(dim, decoder_dim, 1, bias=False),
                        nn.BatchNorm2d(decoder_dim),
                        nn.ReLU(inplace=True)
                    ) for i, dim in enumerate(embed_dims)
                ]
            )
        
        simple_decoder = True
        if simple_decoder == True:
            # simple decoder Light decoder
            self.to_fused = nn.ModuleList(
                [
                nn.Sequential(
                    nn.Conv2d(decoder_dim, decoder_dim, 3, 1, 1),
                    nn.BatchNorm2d(decoder_dim),
                    nn.ReLU(inplace=True)
                ) for i in range(4)
                ] 
            )

            self.to_seg = nn.Conv2d(4 * decoder_dim, decoder_dim, kernel_size=1)
            
        # Classfiyed Layer
        self.to_classify = nn.Conv2d(decoder_dim, 2, kernel_size=1) #

    def forward(self, im1, im2):
        b, c, h, w = im1.shape
        
        x = torch.cat([im1, im2], dim=0)
        
        # backbone mit-b0
        x1, x2, x3, x4 = self.mit(x)
        
        #
        if self.be == True:
            x3 = self.bem[2](x3, x4)
            x2 = self.bem[1](x2, x3)
            x1 = self.bem[0](x1, x2)

        dr = [x1, x2, x3, x4]

        #### simple decoder
        # Dimension Reduction
        for i in range(4):
            dr[i] = self.to_dr[i](dr[i])
        
        x1, x2, x3, x4 = dr
        if self.align == True: # 
            x4 = self.spalign[0](x4, k=3) # 
          
            
            if self.alignv2 == True:
                # deforalign
                x3 = self.dcnalign[0](x3, x4)
                x2 = self.dcnalign[1](x2, x3)

        dr = [x1, x2, x3, x4]

        # difference extraction
        diff = []
        for i in range(4):
            f1, f2 = torch.chunk(dr[i], 2, dim=0)
            diff.append(torch.abs(f1 - f2)) # 
        
        pred = self.seg_head(diff)

        out = [F.interpolate(
            pred,
            (h,w),
            mode="bilinear",
            align_corners=False,
        )]

        return out
    
    #
    def seg_head(self, diff):
        
        x1, x2, x3, x4 = diff

        x2 = F.interpolate(x2, x1.shape[-2:], mode = "bilinear", align_corners = False) #    
        x3 = F.interpolate(x3, x1.shape[-2:], mode = "bilinear", align_corners = False) #   
        x4 = F.interpolate(x4, x1.shape[-2:], mode = "bilinear", align_corners = False) #

        x1 = self.to_fused[0](x1)  
        x2 = self.to_fused[1](x2)
        x3 = self.to_fused[2](x3)
        x4 = self.to_fused[3](x4) # 
        
        x = self.to_seg(torch.cat([x1, x2, x3, x4], dim=1))

        pred = self.to_classify(x)
       
        return pred
    
if __name__ == '__main__':
    pass
