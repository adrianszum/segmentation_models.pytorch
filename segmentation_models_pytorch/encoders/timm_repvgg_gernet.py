from ._base import EncoderMixin
from timm.models.byobnet import ByobNet, default_cfgs, model_cfgs
import torch.nn as nn


class RepVGGEncoder(ByobNet, EncoderMixin):
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
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("head.fc.weight")
        state_dict.pop("head.fc.bias")
        super().load_state_dict(state_dict, **kwargs)


class GENetEncoder(RepVGGEncoder):
    def get_stages(self):
        # join the last two stages in one
        return [
                   nn.Identity(),
                   self.stem] + \
               [s for s in self.stages[:3]] + \
               [nn.Sequential(self.stages[3:])]


timm_repvgg_gernet_encoders = {
    f"timm-{model}": {
        'encoder': RepVGGEncoder if 'vgg' in model else GENetEncoder,
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
            'out_channels': (3, model_cfgs[model].stem_chs) +
                            tuple(int(b.c) for i, b in enumerate(model_cfgs[model].blocks)
                                  if i < 3 or i == (len(model_cfgs[model].blocks) - 1)),
            'cfg': model_cfgs[model]
        }
    }
    for model, cfg in default_cfgs.items()
}
