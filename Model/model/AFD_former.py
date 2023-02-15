## This code is based on the Restormer and NAFNet. Thanks for sharing !
## "Restormer: Efficient Transformer for High-Resolution Image Restoration" (2022 CVPR)
## https://arxiv.org/abs/2111.09881

## NAFNet -> "Simple Baselines for Image Restoration" (2022 ECCV)
## https://arxiv.org/abs/2204.04676


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ptflops import get_model_complexity_info

##########################################################################
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm2d(dim)
        self.attn = Attention(dim, num_heads, bias)

        self.norm2 = LayerNorm2d(dim)
        self.sg = SimpleGate()

        ffn_channel = ffn_expansion_factor * dim
        self.conv_ff1 = nn.Conv2d(in_channels=dim, out_channels=int(ffn_channel), kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv_ff2 = nn.Conv2d(in_channels=int(ffn_channel) // 2, out_channels=dim, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        y = x + self.attn(self.norm1(x)) * self.beta

        x = self.conv_ff1(self.norm2(y))
        x = self.sg(x)
        x = self.conv_ff2(x)

        return y + x * self.gamma


class Localcnn_block(nn.Module):
    def __init__(self, c, DW_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)

        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True))

        self.sg = SimpleGate()

        self.norm1 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        return y


# Transformer and Conv Block
class ConvTransBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, split_conv_rate, bias):
        super(ConvTransBlock, self).__init__()
        self.dim = dim
        self.rate = split_conv_rate

        self.trans_block = TransformerBlock(int(dim - int((dim * split_conv_rate))), num_heads, ffn_expansion_factor, bias)

        self.conv1 = nn.Conv2d(dim, dim, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(dim, dim, 1, 1, 0, bias=True)

        self.conv_block = Localcnn_block(int(dim * split_conv_rate))

    def forward(self, x):
        conv_dim = int(self.dim * self.rate)
        conv_x0, trans_x = torch.split(self.conv1(x), (conv_dim, self.dim - conv_dim), dim=1)

        trans_x = self.trans_block(trans_x)
        conv_x = self.conv_block(conv_x0)

        res = self.conv2(torch.cat((conv_x, trans_x), dim=1))
        out = x + res

        return out


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Sequential(nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##---------- Selective Kernel Feature Fusion (SKFF) ----------
class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)

        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V


##########################################################################
##---------- SKFF_DIBR -----------------------
class AFD_Net(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[2, 6, 8, 10, 4, 3, 2],
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2,
                 rate=[0.8, 0.61, 0.4, 0.23],
                 bias=False):
        super(AFD_Net, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(
            *[ConvTransBlock(dim=int(dim),  num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             split_conv_rate=rate[0], bias=bias) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(
            *[ConvTransBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             split_conv_rate=rate[1], bias=bias) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(
            *[ConvTransBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             split_conv_rate=rate[2], bias=bias) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            ConvTransBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                           split_conv_rate=rate[3], bias=bias) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            ConvTransBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                           split_conv_rate=rate[2], bias=bias) for i in range(num_blocks[4])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *[ConvTransBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             split_conv_rate=rate[1], bias=bias) for i in range(num_blocks[5])])

        self.up2_1 = Upsample(int(dim * 2))  ## From Level 2 to Level 1
        self.decoder_level1 = nn.Sequential(
            *[ConvTransBlock(dim=int(dim * 2), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             split_conv_rate=rate[0], bias=bias) for i in range(num_blocks[6])])

        ####### SKFF #########
        depth = 4

        self.u4_ = nn.Sequential(nn.Conv2d(8 * dim, dim, 3, 1, 1, bias=bias),
                                 nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False))

        self.u3_ = nn.Sequential(nn.Conv2d(4 * dim, dim, 3, 1, 1, bias=bias),
                                 nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False))
        self.u2_ = nn.Sequential(nn.Conv2d(2 * dim, dim, 3, 1, 1, bias=bias),
                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.u1_ = nn.Conv2d(dim * 2, dim, 3, 1, 1, bias=bias)

        self.final_ff = SKFF(in_channels=dim, height=depth)

        self.last = nn.Conv2d(dim, out_channels, 3, 1, 1, bias=True)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)  #
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)  # 2c
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)  # 4c
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        u4_ = self.u4_(latent)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        u3_ = self.u3_(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)  # 2c
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)  # 4c
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)  # 2c
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        u2_ = self.u2_(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)  # c
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)  # 2c
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        u1_ = self.u1_(out_dec_level1)

        skff_in = [u4_, u3_, u2_, u1_]
        skff_out = self.final_ff(skff_in)

        output = self.last(skff_out) + inp_img

        return output


if __name__ == "__main__":

    model = AFD_Net()
    # print(model)

    flops, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)
