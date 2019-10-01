import torch
import torch.nn as nn
import torch.nn.functional as F

class cSE(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.pointwise_1 = nn.Conv2d(in_channels=in_channels, out_channels=(in_channels // 2), kernel_size=1)
        self.pointwise_2 = nn.Conv2d(in_channels=(in_channels // 2), out_channels=in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU6()

    def forward(self, input_tensor):
        pass