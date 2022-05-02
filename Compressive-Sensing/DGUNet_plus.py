import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
# from pdb import set_trace as stx
from torch.nn import init
import numpy as np

##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)  # 计算注意力权重
        y = self.conv_du(y)
        return x * y  # 加权


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


##########################################################################
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias,in_c):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, in_c, kernel_size, bias=bias)
        self.conv3 = conv(in_c, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False, use_HIN=False):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_csff = use_csff

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_dec = nn.Conv2d(in_size, out_size, 3, 1, 1)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size//2, affine=True)
        self.use_HIN = use_HIN

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, enc=None, dec=None):
        out = self.conv_1(x)

        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        out += self.identity(x)
        if enc is not None and dec is not None:
            assert self.use_csff
#             print('enc+dec:',out.shape,enc.shape,dec.shape)
            out = out + self.csff_enc(enc) + self.csff_dec(dec)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(out_size*2, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out

##########################################################################
## U-Net

class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff,depth=4):
        super(Encoder, self).__init__()
        self.body=nn.ModuleList()#[]
        self.depth=depth
        for i in range(depth-1):
            self.body.append(UNetConvBlock(in_size=n_feat+scale_unetfeats*i, out_size=n_feat+scale_unetfeats*(i+1), downsample=True, relu_slope=0.2, use_csff=True, use_HIN=True))
        
        self.body.append(UNetConvBlock(in_size=n_feat+scale_unetfeats*(depth-1), out_size=n_feat+scale_unetfeats*(depth-1), downsample=False, relu_slope=0.2, use_csff=True, use_HIN=True))

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        res=[]
        if encoder_outs is not None and decoder_outs is not None:
            for i,down in enumerate(self.body):
                if (i+1) < self.depth:
                    x, x_up = down(x,encoder_outs[i],decoder_outs[-i-1])
                    res.append(x_up)
                else:
                    x = down(x)
        else:
            for i,down in enumerate(self.body):
                if (i+1) < self.depth:
                    x, x_up = down(x)
                    res.append(x_up)
                else:
                    x = down(x)
        return res,x


class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=4):
        super(Decoder, self).__init__()
        
        self.body=nn.ModuleList()
        self.skip_conv=nn.ModuleList()#[]
        for i in range(depth-1):
            self.body.append(UNetUpBlock(in_size=n_feat+scale_unetfeats*(depth-i-1), out_size=n_feat+scale_unetfeats*(depth-i-2), relu_slope=0.2))
            self.skip_conv.append(nn.Conv2d(n_feat+scale_unetfeats*(depth-i-1), n_feat+scale_unetfeats*(depth-i-2), 3, 1, 1))
            
    def forward(self, x, bridges):
        res=[]
        for i,up in enumerate(self.body):
            x=up(x,self.skip_conv[i](bridges[-i-1]))
            res.append(x)

        return res


##########################################################################
##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


##########################################################################
## Original Resolution Block (ORB)
class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


##########################################################################
class ORSNet(nn.Module):
    def __init__(self, n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):
        super(ORSNet, self).__init__()

        self.orb1 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat+2*scale_unetfeats, n_feat+scale_unetfeats)
        self.up_dec1 = UpSample(n_feat+scale_unetfeats, n_feat)

        self.up_enc2 = nn.Sequential(UpSample(n_feat + 3*scale_unetfeats, n_feat+2*scale_unetfeats),
                                     UpSample(n_feat+2*scale_unetfeats, n_feat+scale_unetfeats))
        self.up_dec2 = nn.Sequential(UpSample(n_feat + 2*scale_unetfeats, n_feat+scale_unetfeats),
                                     UpSample(n_feat+scale_unetfeats, n_feat))
        

        self.conv_enc1 = nn.Conv2d(n_feat+scale_unetfeats, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc2 = nn.Conv2d(n_feat+scale_unetfeats, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc3 = nn.Conv2d(n_feat+scale_unetfeats, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)

        self.conv_dec1 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[-1])

        x = self.orb2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[-2]))

        x = self.orb3(x)
        x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[-3]))

        return x

def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2),stride=stride, bias=bias)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.PReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            if i == 0:
                m.append(conv(n_feats, 64, kernel_size, bias=bias))
            else:
                m.append(conv(64, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

##########################################################################
class MPRBlock(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=80, scale_unetfeats=48, kernel_size=3,
                 reduction=4, bias=False):
        super(MPRBlock, self).__init__()
        act = nn.PReLU()
        self.shallow_feat = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))

        # Cross Stage Feature Fusion (CSFF)
        self.stage_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)
        self.stage_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.sam = SAM(n_feat, kernel_size=1, bias=bias,in_c=in_c)

        self.phi = ResBlock(default_conv, 3, 3)
        self.phit = ResBlock(default_conv, 3, 3)

        self.r = nn.Parameter(torch.Tensor([0.5]))

        self.concat = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
    def forward(self, stage_img, img,x_samfeats,f_encoder,f_decoder,PhiTPhi,PhiTb):
        b,c,w,h = stage_img.shape
        x_k_1=stage_img.view(b,-1)
        # compute r_k
        x = x_k_1 - self.r * torch.mm(x_k_1, PhiTPhi)
        r_k = x + self.r * PhiTb
        r_k = r_k.view(b,c,w,h)

        # compute x_k
        x = self.shallow_feat(r_k)
        ## Concatenate SAM features of Stage 1 with shallow features of Stage 2
        x_cat = self.concat(torch.cat([x, x_samfeats], 1))
        ## Process features of both patches with Encoder of Stage 2
        feat1, f_encoder = self.stage_encoder(x_cat, f_encoder, f_decoder)
        ## Pass features through Decoder of Stage 2
        f_decoder = self.stage_decoder(f_encoder, feat1)
        ## Apply SAM
        x_samfeats, stage_img = self.sam(f_decoder[-1], r_k)
        return stage_img, x_samfeats,feat1 ,f_decoder

##########################################################################
class DGUNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=32, scale_unetfeats=16, scale_orsnetfeats=16, num_cab=8, kernel_size=3,
                 reduction=4, bias=False, nums_stages=5, cs_ratio=25):
        super(DGUNet, self).__init__()
        print('CS Ratio: ',cs_ratio)

        self.Phi = nn.Parameter(init.xavier_normal_(torch.Tensor(np.ceil(cs_ratio*0.01*1024.).astype(np.int), 1024)))
        act = nn.PReLU()
        self.shallow_feat1 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.body = nn.ModuleList()
        for _ in range(nums_stages):
            self.body.append(MPRBlock(
                in_c=in_c, out_c=in_c, n_feat=n_feat, scale_unetfeats=scale_unetfeats, kernel_size=kernel_size,
                reduction=reduction, bias=bias
            ))
        self.shallow_feat_final = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))

        # Cross Stage Feature Fusion (CSFF)
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias,in_c=in_c)

        self.r0 = nn.Parameter(torch.Tensor([0.5]))
        self.r_final = nn.Parameter(torch.Tensor([0.5]))

        self.concat_final = conv(n_feat * 2, n_feat + scale_orsnetfeats, kernel_size, bias=bias)
        self.tail = conv(n_feat + scale_orsnetfeats, out_c, kernel_size, bias=bias)

    def forward(self, img):
        output_=[]
        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        b, c, w, h = img.shape
        PhiTPhi = torch.mm(torch.transpose(self.Phi, 0, 1), self.Phi)  # torch.mm(Phix, Phi)
        Phix = torch.mm(img.view(b,-1), torch.transpose(self.Phi, 0, 1))  # compression result
        PhiTb = torch.mm(Phix,self.Phi)
        # compute r_0
        x_0=PhiTb.view(b,-1)
        x = x_0 - self.r0 * torch.mm(x_0, PhiTPhi)
        r_0 = x + self.r0 * PhiTb
        r_0=r_0.view(b,c,w,h)
        
        # compute x_k
        x = self.shallow_feat1(r_0)
        ## Process features of all 4 patches with Encoder of Stage 1
        feat1, f_encoder = self.stage1_encoder(x)
        ## Pass features through Decoder of Stage 1
        f_decoder = self.stage1_decoder(f_encoder,feat1)
        ## Apply Supervised Attention Module (SAM)
        x_samfeats, stage_img = self.sam12(f_decoder[-1], r_0)
        output_.append(stage_img)

        ##-------------------------------------------
        ##-------------- Stage 2_k-1---------------------
        ##-------------------------------------------
        for stage_model in self.body: 
            stage_img, x_samfeats, feat1, f_decoder = stage_model(stage_img, img,x_samfeats,feat1,f_decoder,PhiTPhi,PhiTb)
            output_.append(stage_img)
        ##-------------------------------------------
        ##-------------- Stage k---------------------
        ##-------------------------------------------
        # compute r_k
        x_k_1 = stage_img.view(b,-1)
        x = x_k_1 - self.r_final * torch.mm(x_k_1, PhiTPhi)
        r_k = x + self.r_final * PhiTb
        r_k = r_k.view(b,c,w,h)
        
        # compute x_k
        x = self.shallow_feat_final(r_k)
        ## Concatenate SAM features
        x_cat = self.concat_final(torch.cat([x, x_samfeats], 1))
        stage_img = self.tail(x_cat)+r_k
        output_.append(stage_img)

        return output_[::-1]
