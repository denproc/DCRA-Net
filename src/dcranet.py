# MIT License
#
# Copyright (c) 2024 Denis Prokopenko

import torch
from torch import nn
from functools import partial
from rotary_embedding_torch import RotaryEmbedding
from src.dataconsistency import DataConsistencyKSpace
from src.transforms import ToImage, ToFrequency, ToKSpace, ToTime

from video_diffusion_pytorch.video_diffusion_pytorch import (
    EinopsToAndFrom,
    Attention,
    RelativePositionBias,
    Residual,
    PreNorm,
    SpatialLinearAttention,
    Block,
    Downsample,
    Upsample,
    default,
    is_odd,
)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


class UNet3D(nn.Module):
    def __init__(
        self,
        dim,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        attn_heads=8,
        attn_dim_head=32,
        init_dim=None,
        init_kernel_size=7,
        use_sparse_linear_attn=True,
        resnet_groups=8,
    ):
        super().__init__()
        self.channels = channels

        # temporal attention and its relative positional encoding

        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))

        def temporal_attn(dim):
            return EinopsToAndFrom(
                "b c f h w",
                "b (h w) f c",
                Attention(
                    dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb
                ),
            )

        self.time_rel_pos_bias = RelativePositionBias(heads=attn_heads, max_distance=32)

        # initial conv

        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(
            channels,
            init_dim,
            (1, init_kernel_size, init_kernel_size),
            padding=(0, init_padding, init_padding),
        )

        self.init_temporal_attn = Residual(PreNorm(init_dim, temporal_attn(init_dim)))

        # dimensions

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        # block type

        block_klass = partial(ResnetBlock, groups=resnet_groups)

        # modules for all layers

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out),
                        block_klass(dim_out, dim_out),
                        (
                            Residual(
                                PreNorm(
                                    dim_out,
                                    SpatialLinearAttention(dim_out, heads=attn_heads),
                                )
                            )
                            if use_sparse_linear_attn
                            else nn.Identity()
                        ),
                        Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom(
            "b c f h w", "b f (h w) c", Attention(mid_dim, heads=attn_heads)
        )

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))

        self.mid_block2 = block_klass(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in),
                        block_klass(dim_in, dim_in),
                        (
                            Residual(
                                PreNorm(
                                    dim_in,
                                    SpatialLinearAttention(dim_in, heads=attn_heads),
                                )
                            )
                            if use_sparse_linear_attn
                            else nn.Identity()
                        ),
                        Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim), nn.Conv3d(dim, out_dim, 1)
        )

    def forward(self, x):

        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)

        x = self.init_conv(x)

        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)

        r = x.clone()

        h = []

        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x)
            x = block2(x)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x, pos_bias=time_rel_pos_bias)
        x = self.mid_block2(x)

        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x)
            x = block2(x)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)


class DCRANet(nn.Module):
    def __init__(
        self,
        dim,
        out_dim=None,
        dim_mults=(1, 2, 4),
        channels=2,
        attn_heads=8,
        attn_dim_head=32,
        init_dim=None,
        init_kernel_size=7,
        use_sparse_linear_attn=True,
        resnet_groups=8,
        dc_mode="",
        norm_fft=None,
        representation_time="frequency",
    ):
        super().__init__()

        self.unet3d = UNet3D(
            dim=dim,
            out_dim=out_dim,
            dim_mults=dim_mults,
            attn_dim_head=attn_dim_head,
            init_dim=init_dim,
            channels=channels,
            attn_heads=attn_heads,
            init_kernel_size=init_kernel_size,
            use_sparse_linear_attn=use_sparse_linear_attn,
            resnet_groups=resnet_groups,
        )
        self.representation_time = representation_time
        self.dc = DataConsistencyKSpace(dc_mode=dc_mode)
        self.norm = norm_fft

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # N, 1, D, H, 1
        assert x.size(1) == 1, f"Expected 1channel k-space, got {x.size()}"
        k_mask = (x.abs().sum(dim=-1, keepdim=True) > 0).type(torch.float)

        # (N, 1, D, H, W) ->  (N, D, H, W, 2) ->  (N, 2, D, H, W)
        image_input = ToImage(norm=self.norm)(x)
        if self.representation_time == "frequency":
            image_input = ToFrequency(norm=self.norm)(image_input)
        elif self.representation_time == "time":
            pass
        else:
            raise ValueError(
                f"representation_time should be in [time, frequency], got {self.representation_time}"
            )

        image_input = torch.view_as_real(image_input.squeeze(dim=1)).type(torch.float)
        image_input = image_input.permute(0, 4, 1, 2, 3)
        image_prediction = self.unet3d(image_input)

        # (N, 2, D,  H, W)-> (N, D, H, W, 2) -> complex (N, D, H, W)
        # -> complex (N, 1, D, H, W)
        complex_image_pred = image_prediction.permute(0, 2, 3, 4, 1)
        complex_image_pred = complex_image_pred.contiguous()
        complex_image_pred = torch.view_as_complex(complex_image_pred)
        complex_image_pred = complex_image_pred.unsqueeze(1)
        # (N, 1, D, H, W)
        kspace_prediction = ToKSpace(norm=self.norm)(complex_image_pred)
        if self.representation_time == "frequency":
            kspace_prediction = ToTime(norm=self.norm)(kspace_prediction)
        elif self.representation_time == "time":
            pass
        else:
            raise ValueError(
                f"representation_time should be in [time, frequency], got {self.representation_time}"
            )

        kspace_out = self.dc(kspace_prediction, x, k_mask)  # (N, 1, D, H, W)

        # complex (N, 1, H, W) -> complex (N, H, W) -> real (N, H, W , 2)
        # -> real (N, 2, H, W)
        image_out = ToImage(norm=self.norm)(kspace_out).squeeze(dim=1)
        if self.representation_time == "frequency":
            image_out = ToFrequency(norm=self.norm)(image_out)
        elif self.representation_time == "time":
            pass
        else:
            raise ValueError(
                f"representation_time should be in [time, frequency], got {self.representation_time}"
            )
        image_out = torch.view_as_real(image_out).permute(0, 4, 1, 2, 3)
        return image_out
