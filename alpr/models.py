import torch
from torch import nn
from .common import ALPHABET


def conv(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Depthwise(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
            nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
            nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layers(x)


class Detector(nn.Module):
    """Stride-8 centre detector; context branches have stride 16 and 32."""
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(conv(3, 24, 2), Depthwise(24, 40, 2),
                                  Depthwise(40, 64, 2), Depthwise(64, 64))
        self.context = nn.Sequential(Depthwise(64, 96, 2), Depthwise(96, 128, 2), Depthwise(128, 128))
        self.fuse = nn.Sequential(conv(192, 96), Depthwise(96, 96))
        self.heat = nn.Conv2d(96, 4, 1)
        self.reg = nn.Conv2d(96, 13, 1)  # centre offset(2), size(2), quad offsets(8), vehicle(1)
        nn.init.constant_(self.heat.bias, -2.19)

    def forward(self, x):
        a = self.base(x)
        b = nn.functional.interpolate(self.context(a), size=a.shape[-2:], mode='bilinear', align_corners=False)
        f = self.fuse(torch.cat([a, b], dim=1))
        return self.heat(f), self.reg(f)


class Recognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(conv(1, 32, 2), conv(32, 64, 2),
                                      Depthwise(64, 96), Depthwise(96, 128),
                                      nn.AdaptiveAvgPool2d((1, 80)))
        self.sequence = nn.GRU(128, 96, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(192, len(ALPHABET) + 1)  # zero = CTC blank

    def forward(self, x):
        f = self.features(x).squeeze(2).transpose(1, 2)
        f, _ = self.sequence(f)
        return self.classifier(f)  # N,T,C


def load_model(path, expected, device='cpu'):
    # Only tensors/primitives are accepted. No arbitrary pickle execution.
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if ckpt.get('kind') != expected or ckpt.get('version') != 1:
        raise ValueError(f'Wrong checkpoint format for {expected}: {path}')
    if ckpt.get('types') != ['type1', 'type1a', 'type1b', 'other'] or ckpt.get('alphabet') != ALPHABET:
        raise ValueError('Checkpoint alphabet/types do not match the code')
    net = Detector() if expected == 'detector' else Recognizer()
    net.load_state_dict(ckpt['state_dict'])
    return net.to(device).eval(), ckpt


def save_model(path, model, kind, **info):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'version': 1, 'kind': kind, 'types': ['type1', 'type1a', 'type1b', 'other'],
                'alphabet': ALPHABET, 'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                **info}, path)
