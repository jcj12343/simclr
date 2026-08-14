import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class SimCLR(nn.Module):
    def __init__(self, encoder_out_dim=512, proj_hidden=512, proj_out_dim=128):
        super().__init__()
        self.encoder = resnet18(pretrained=False)
        # cifar‑10 32×32适配
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.maxpool = nn.Identity()
        self.encoder.fc = nn.Identity()
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_out_dim, proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_out_dim)
        )
    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        z = F.normalize(z, dim=1)
        return h, z

class LinearProbeModel(nn.Module):
    def __init__(self, encoder, num_classes=10):
        super().__init__()
        self.encoder = encoder
        self.encoder.requires_grad_(False)
        self.linear = nn.Linear(512, num_classes)
    def forward(self, x):
        feat = self.encoder(x)
        return self.linear(feat)