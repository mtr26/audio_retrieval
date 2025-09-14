import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

"""
Based on arxiv:2010.11910
"""

@dataclass
class ModelConfig:
    d_model: int = 64
    hidden_size: int = 1024
    u_size: int = 32,
    batch_size: int = 32
    model_path: str = None
    compile_mode: str = 'default'  # 'default', 'reduce_overhead', 'max_autotune', 'min_autotune' or 'None'
    device: str = 'cpu'  # 'cpu' or 'cuda', for tpu load using torch_xla, but multiprocessing on tpu not tested yet

class SeperableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super(SeperableConv2d, self).__init__()
        pad_h = kernel_size // 2
        pad_w = kernel_size // 2
        padding_horiz = (0, pad_w)
        padding_vert = (pad_h, 0)
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=(1, kernel_size),
                               stride=stride[0],
                               padding=padding_horiz)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=(kernel_size, 1),
                               stride=stride[1],
                               padding=padding_vert)
        self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        return out


class ConvolutionalEncoder(nn.Module):
    def __init__(self, in_channels, d_model, kernel_size, stride, h):
        super(ConvolutionalEncoder, self).__init__()
        layers_params = [
            (in_channels, d_model, [(1,2), (2,1)]), # 1
            (d_model, d_model, [(1,2), (2,1)]), # 2
            (d_model, 2*d_model, [(1,2), (2,1)]), # 3
            (2*d_model, 2*d_model, [(1,2), (2,1)]), # 4
            (2*d_model, 4*d_model, [(1,1), (2,1)]), # 5
            (4*d_model, 4*d_model, [(1,2), (2,1)]), # 6
            (4*d_model, h, [(1,1), (2,1)]), # 7
            (h, h, [(1,2), (2,1)]) # 8
        ]
        self.layers = nn.ModuleList([
            SeperableConv2d(
                in_channels=i,
                out_channels=o,
                kernel_size=kernel_size,
                stride=s
            )
            for i, o, s in layers_params
        ])
        self.flat = nn.Flatten()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.flat(x)
        return x


class ProjectionHead(nn.Module):
    def __init__(self, h, d, u):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(h, u),
            nn.ELU(inplace=True),
            nn.Linear(u, d)
        )

    def forward(self, x):
        y = self.proj(x)
        y = F.normalize(y, p=2, dim=1)  # final embedding normalized
        return y


class FingerPrintModel(nn.Module):
    def __init__(self, d_model=64, hidden_size=1024, u_size=32):
        super(FingerPrintModel, self).__init__()
        self.encoder = ConvolutionalEncoder(
            in_channels=1,
            d_model=d_model,
            kernel_size=3,
            stride=2,
            h=hidden_size
        )
        self.projection_head = ProjectionHead(
            h=hidden_size,
            d=d_model,
            u=u_size
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.projection_head(x)
        return x