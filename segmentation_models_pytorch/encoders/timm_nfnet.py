from ._base import EncoderMixin
from timm.models.nfnet import NormFreeNet, default_cfgs, model_cfgs
import torch.nn as nn


class NFNetEncoder(NormFreeNet, EncoderMixin):
    def __init__(self, out_channels, depth: int = 5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3
        del self.head

    def get_stages(self):
        return [
                   nn.Identity(),
                   self.stem] + \
               [s for s in self.stages]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            if i > 0:
                # Some NFNets have 4x downsampling in stem
                # We can upsample 2x this feature map in order not to dig deeper
                reduction = features[-1].shape[-1] // x.shape[-1]
                # print(reduction, reduction // 2, features[-1].shape[-1], x.shape[-1])
                if reduction > 2:
                    upsampled = nn.functional.interpolate(x, scale_factor=reduction // 2, mode='nearest')
                    features.append(upsampled)
                    continue
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("head.fc.weight")
        state_dict.pop("head.fc.bias")
        super().load_state_dict(state_dict, **kwargs)


nfnet_models_out_channels = {
    'nf_regnet_b0': (3, 40, 40, 80, 160, 328),
    'nf_regnet_b1': (3, 40, 40, 80, 160, 328),  # pretrained
    'nf_regnet_b2': (3, 40, 40, 88, 176, 368),
    'nf_regnet_b3': (3, 40, 40, 96, 184, 400),
    'nf_regnet_b4': (3, 48, 48, 112, 216, 464),
    'nf_regnet_b5': (3, 64, 64, 128, 256, 528),
    # other parameters are correct in cfg, keep them just in case
    'nf_resnet50': (3, 64, 256, 512, 1024, 2048),
    'nfnet_l0c': (3, 128, 256, 512, 1536, 1536),
    'dm_nfnet_f0': (3, 128, 256, 512, 1536, 1536),
    'dm_nfnet_f1': (3, 128, 256, 512, 1536, 1536),
    'dm_nfnet_f2': (3, 128, 256, 512, 1536, 1536),
    'dm_nfnet_f3': (3, 128, 256, 512, 1536, 1536),
    'dm_nfnet_f4': (3, 128, 256, 512, 1536, 1536),
    'dm_nfnet_f5': (3, 128, 256, 512, 1536, 1536),
    'dm_nfnet_f6': (3, 128, 256, 512, 1536, 1536),
}

timm_nfnet_encoders = {
    f"timm-{model}": {
        'encoder': NFNetEncoder,
        'pretrained_settings': {
            'imagenet': {
                'url': cfg['url'],
                'input_size': cfg['input_size'],
                'input_range': [0, 1],
                'mean': cfg['mean'],
                'std': cfg['std'],
                'num_classes': cfg['num_classes']
            }
        } if cfg['url'] else {},
        'params': {
            # regnets have cgf.channels that do not match the actual numbers and empty cfg.stem_chs
            'out_channels': nfnet_models_out_channels[model] if 'regnet' in model
            else (3, model_cfgs[model].stem_chs) + model_cfgs[model].channels,
            'cfg': model_cfgs[model]
        }
    }
    for model, cfg in default_cfgs.items()
}
