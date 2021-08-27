from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_

from Transformer.swin_transformer3d import PatchEmbed3D, BasicLayer


class SingleConv(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=3, stride=1, padding=1):
        super(SingleConv, self).__init__()

        self.single_conv = nn.Sequential(OrderedDict([
            ('conv3d',
             nn.Conv3d(in_chans, out_chans, kernel_size=kernel_size, padding=padding, stride=stride, bias=True)),
            ('InsNorm', nn.InstanceNorm3d(out_chans, affine=True)),
            ('relu', nn.ReLU(inplace=True))
        ]))

    def forward(self, x):
        return self.single_conv(x)


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor=2):
        super(UpConv, self).__init__()
        self.scale_factor = scale_factor

        self.conv = SingleConv(in_chans=in_ch, out_chans=out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='trilinear', align_corners=True)
        x = self.conv(x)
        return x


class PatchExpanding(nn.Module):
    """
    Patch Expanding layer
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super(PatchExpanding, self).__init__()
        self.dim = dim
        self.expanding = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        pass


class PatchMerging(nn.Module):
    """ Patch Merging Layer, modified from Video Swin Transformer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        # self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)

        # downsampling using conv
        # self.reduction = SingleConv(in_chans=dim, out_chans=2 * dim, kernel_size=3, stride=2, padding=1)

        # downsampling using pooling
        self.reduction = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            SingleConv(in_chans=dim, out_chans=2 * dim, kernel_size=3, stride=1, padding=1)
        )

        # self.norm = norm_layer(8 * dim)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        # x0 = x[:, 0::2, 0::2, 0::2, :]  # B D/2 H/2 W/2 C
        # x1 = x[:, 0::2, 1::2, 0::2, :]  # B D/2 H/2 W/2 C
        # x2 = x[:, 0::2, 0::2, 1::2, :]  # B D/2 H/2 W/2 C
        # x3 = x[:, 0::2, 1::2, 1::2, :]  # B D/2 H/2 W/2 C
        # x4 = x[:, 1::2, 0::2, 0::2, :]  # B D/2 H/2 W/2 C
        # x5 = x[:, 1::2, 1::2, 0::2, :]  # B D/2 H/2 W/2 C
        # x6 = x[:, 1::2, 0::2, 1::2, :]  # B D/2 H/2 W/2 C
        # x7 = x[:, 1::2, 1::2, 1::2, :]  # B D/2 H/2 W/2 C
        # x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B D/2 H/2 W/2 8*C

        x = rearrange(x, 'b d h w c -> b c d h w')
        # downsampling using interpolation
        # x = F.interpolate(x, scale_factor=0.5, mode='trilinear', align_corners=True)
        # x = SingleConv(in_chans=self.dim, out_chans=self.dim * 2, kernel_size=3, stride=1, padding=1)(x)

        # x = self.norm(x)
        x = self.reduction(x)
        x = rearrange(x, 'b c d h w -> b d h w c')

        return x


class PatchConv3D(nn.Module):
    """
    replace PatchEmbed3D by Conv3d stem to to increase optimization stability.

    Xiao, T., Singh, M., Mintun, E., Darrell, T., Dollár, P., & Girshick, R. (2021).
    Early Convolutions Help Transformers See Better. 1–15. http://arxiv.org/abs/2106.14881
    """

    def __init__(self, patch_size=(2, 4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(OrderedDict([
            ('downsampling_1', SingleConv(in_chans, int(embed_dim / 4), kernel_size=3, stride=2, padding=1)),  # 24
            ('downsampling_2', SingleConv(int(embed_dim / 4), int(embed_dim / 2), kernel_size=3, stride=2, padding=1)),
            # 48
            # ('conv1x1', nn.Conv3d(int(embed_dim / 2), embed_dim, kernel_size=1, stride=1, padding=0)),  # 96
            ('conv3x3', SingleConv(int(embed_dim / 2), embed_dim, kernel_size=3, stride=1, padding=1)),
            ('conv1x1', nn.Conv3d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0))
        ]))

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x


class SwinTU3D(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(self,
                 pretrained=None,
                 patch_size=(2, 4, 4),
                 conv_stem=True,
                 in_chans=3,
                 embed_dim=96,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 window_size=(8, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrained = pretrained
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size
        self.conv_stem = conv_stem

        # split image into non-overlapping patches
        # self.patch_embed = PatchEmbed3D(
        #     patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
        #     norm_layer=norm_layer if self.patch_norm else None)
        # self.patch_embed = PatchConv3D(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
        #                                norm_layer=norm_layer if self.patch_norm else None)
        if conv_stem:
            self.patch_embed = PatchConv3D(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                                           norm_layer=norm_layer if self.patch_norm else None)
        else:
            self.patch_embed = PatchEmbed3D(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                                            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # build decoders
        self.decoders = nn.ModuleList()
        for i_decoder in range(self.num_layers - 2, -1, -1):  # (2, 1, 0)
            decoder = BasicLayer(
                dim=int(embed_dim * 2 ** i_decoder),
                depth=depths[i_decoder],
                num_heads=num_heads[i_decoder],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_decoder]):sum(depths[:i_decoder + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint)
            self.decoders.append(decoder)
        self.decoders = self.decoders[::-1]

        # build Upsample layers
        self.up_layers = nn.ModuleList()
        for i in range(self.num_layers - 2, -1, -1):
            up_layer = UpConv(in_ch=int(embed_dim * 2 ** (i + 1)), out_ch=int(embed_dim * 2 ** i),
                              scale_factor=2)
            self.up_layers.append(up_layer)
        self.up_layers = self.up_layers[::-1]

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # fusion
        self.fusions = nn.ModuleList()
        for i in range(self.num_layers - 1):  # [0, 1, 2]
            fusion = SingleConv(in_chans=2 * num_features[i], out_chans=num_features[i], kernel_size=3, stride=1)
            self.fusions.append(fusion)

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        # self.upsample_2 = UpConv(in_ch=self.num_features[0], out_ch=int(self.num_features[0] / 2), scale_factor=2)
        # self.upsample_1 = UpConv(in_ch=int(self.num_features[0] / 2), out_ch=int(self.num_features[0] / 4),
        #                          scale_factor=2)
        # self.conv_out = SingleConv(in_chans=int(self.num_features[0] / 4), out_chans=1, kernel_size=3)

        self.head = nn.Sequential(OrderedDict([
            ('upsampling_2', UpConv(in_ch=self.num_features[0], out_ch=int(self.num_features[0] / 2), scale_factor=2)),
            ('upsampling_1', UpConv(in_ch=int(self.num_features[0] / 2), out_ch=int(self.num_features[0] / 4),
                                    scale_factor=2)),
            # conv out
            ('conv3x3', SingleConv(in_chans=int(self.num_features[0] / 4), out_chans=1, kernel_size=3)),
            ('conv1x1', nn.Conv3d(in_channels=1, out_channels=1, kernel_size=1))
        ]))

        # for fine tuning
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            # added by Chenchen Hu
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.InstanceNorm3d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(pretrained, str):
            self.apply(_init_weights)
            # logger = get_root_logger()
            # load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)  # B C D Wh Ww

        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, x = layer(x)  # x is the output of PatchMerging

            if i in self.out_indices:
                x_out = rearrange(x_out, 'n c d h w -> n d h w c')
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = rearrange(x_out, 'n d h w c -> n c d h w')
                outs.append(out)

        y = outs[-1]
        y_out = outs.pop()
        for i in range(1, self.num_layers):  # [1, 2, 3]
            decoder = self.decoders[-i]
            up_layer = self.up_layers[-i]
            y = up_layer(y)
            # add skip connection here
            shortcut = outs[-i]
            y = torch.cat([y, shortcut], dim=1)
            y = self.fusions[-i](y)
            y_out, y = decoder(y)

        # y_out = self.upsample_2(y_out)  # embed_dim -> embed_dim / 2  48
        # y_out = self.upsample_1(y_out)  # 24
        # y_out = self.conv_out(y_out)
        y_out = self.head(y_out)

        return y_out


class DualStemModel(nn.Module):
    def __init__(self, knowledge_branch=None):
        super(DualStemModel, self).__init__()
        self.CT_branch = SwinTU3D(patch_size=(4, 4, 4), depths=(2, 2, 6, 2), norm_layer=torch.nn.LayerNorm)
        self.knowledge_branch = SwinTU3D(patch_size=(4, 4, 4), depths=[2, 2], num_heads=[3, 6], out_indices=[0, 1],
                                         in_chans=1, conv_stem=False)
        self.conv1x1 = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_knowledge = x[:, -1, :, :, :]
        x_knowledge = torch.unsqueeze(x_knowledge, dim=1)
        x_CT = x[:, :-1, :, :, :]

        y = self.CT_branch(x_CT) + self.knowledge_branch(x_knowledge)
        y = self.conv1x1(y)

        return y
