from ._base import EncoderMixin
from timm.models.swin_transformer import SwinTransformer, default_cfgs, _cfg
import torch.nn as nn
from math import sqrt
from einops import rearrange


class SwinEncoder(SwinTransformer, EncoderMixin):
    def __init__(self, out_channels, depth: int = 5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3
        del self.head

    def get_stages(self):
        return [nn.Identity(),
                self.patch_embed,
                nn.Sequential(self.pos_drop, self.layers[0]),
                self.layers[1],
                self.layers[2],
                self.layers[3]]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            # first stage is identity
            if i == 0:
                features.append(x)
                continue
            # next ones are flattened from transformer starting from (h/4 w/4)
            r = rearrange(x, "b (h w) c -> b c h w", h=int(sqrt(x.shape[1])))
            # do not interpolate the last stage, no downsampling there
            if i != self._depth:
                r = nn.functional.interpolate(r, scale_factor=2, mode='nearest')
            # we can add positional embeddings but after feature construction
            if i == 1 and self.absolute_pos_embed is not None:
                x = x + self.absolute_pos_embed

            features.append(r)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        model_state_dict = state_dict['model']
        model_state_dict.pop("head.weight")
        model_state_dict.pop("head.bias")
        super().load_state_dict(model_state_dict, **kwargs)


# swin_out_channels = {
#     "tiny": 96,
#     "small": 96,
#     "base": 128,
#     "large": 192,
# }
#
# swin_out_channels = {s: (3, ) + tuple(d * 2 ** min(i, 3) for i in range(5))
#                      for s, d in swin_out_channels.items()}

timm_swin_encoders = {
    "timm-swin_base_patch4_window12_384": {
        "encoder": SwinEncoder,
        "pretrained_settings": {
            "imagenet": _cfg(url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth',
                             input_size=(3, 384, 384), crop_pct=1.0)
        },
        "params": dict(
            out_channels=(3, 128, 256, 512, 1024, 1024), img_size=384, crop_pct=1.0,
            patch_size=4, window_size=12, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
    },
    "timm-swin_base_patch4_window7_224": {
        "encoder": SwinEncoder,
        "pretrained_settings": {
            "imagenet": _cfg(url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth'),
        },
        "params": dict(
            out_channels=(3, 128, 256, 512, 1024, 1024), img_size=224,
            patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
    },
    "timm-swin_large_patch4_window12_384": {
        "encoder": SwinEncoder,
        "pretrained_settings": {
            "imagenet": _cfg(url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth'),
        },
        "params": dict(
            out_channels=(3, 192, 384, 768, 1536, 1536), img_size=384, crop_pct=1.0,
            patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    },
    "timm-swin_large_patch4_window7_224": {
        "encoder": SwinEncoder,
        "pretrained_settings": {
            "imagenet": _cfg(url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22kto1k.pth'),
        },
        "params": dict(
            out_channels=(3, 192, 384, 768, 1536, 1536), img_size=224,
            patch_size=4, window_size=7, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    },
    "timm-swin_small_patch4_window7_224": {
        "encoder": SwinEncoder,
        "pretrained_settings": {
            "imagenet": _cfg(url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'),
        },
        "params": dict(
            out_channels=(3, 96, 192, 384, 768, 768), img_size=224,
            patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24))
    },
    "timm-swin_tiny_patch4_window7_224": {
        "encoder": SwinEncoder,
        "pretrained_settings": {
            "imagenet": _cfg(url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'),
        },
        "params": dict(
            out_channels=(3, 96, 192, 384, 768, 768), img_size=224,
            patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24))
    },
    "timm-swin_base_patch4_window12_384_in22k": {
        "encoder": SwinEncoder,
        "pretrained_settings": {
            "imagenet": _cfg(url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth'),
        },
        "params": dict(
            out_channels=(3, 128, 256, 512, 1024, 1024), img_size=384, crop_pct=1.0,
            patch_size=4, window_size=12, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
    },
    "timm-swin_base_patch4_window7_224_in22k": {
        "encoder": SwinEncoder,
        "pretrained_settings": {
            "imagenet": _cfg(url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth'),
        },
        "params": dict(
            out_channels=(3, 128, 256, 512, 1024, 1024), img_size=224,  num_classes=21841,
            patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
    },
    "timm-swin_large_patch4_window12_384_in22k": {
        "encoder": SwinEncoder,
        "pretrained_settings": {
            "imagenet": _cfg(url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'),
        },
        "params": dict(
            out_channels=(3, 192, 384, 768, 1536, 1536), img_size=384, crop_pct=1.0, num_classes=21841,
            patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    },
    "timm-swin_large_patch4_window7_224_in22k": {
        "encoder": SwinEncoder,
        "pretrained_settings": {
            "imagenet": _cfg(url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth'),
        },
        "params": dict(
            out_channels=(3, 192, 384, 768, 1536, 1536), img_size=224, num_classes=21841,
            patch_size=4, window_size=7, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    },
}

# timm_swin_encoders = {
#     f"timm-{model}": {
#         'encoder': SwinEncoder,
#         'pretrained_settings': {
#             'imagenet': {
#                 'url': cfg['url'],
#                 'input_size': cfg['input_size'],
#                 'input_range': [0, 1],
#                 'mean': cfg['mean'],
#                 'std': cfg['std'],
#                 'num_classes': cfg['num_classes']
#             }
#         } if cfg['url'] else {},
#         'params': {
#             'out_channels': swin_out_channels[model.split('_')[1]]
#         }
#     }
#     for model, cfg in default_cfgs
# }