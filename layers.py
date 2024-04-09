import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, feature_act = nn.ReLU()):
        super(GatedConv, self).__init__()
        
        self.gating = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.feature = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        
        # normalizes inputs, improves the generalization ability of the model
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        
        self.gating_act = nn.Sigmoid()
        self.feature_act = feature_act
        
    def forward(self, x):
        gating = self.gating(x)
        feature = self.feature(x)
        
        if self.feature_act is None:
            out = self.gating_act(gating) * feature
        else:    
            out = self.gating_act(gating) * self.feature_act(feature)
        
        out = self.batch_norm(out)
        return out
        

class ResizeGatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor = 2):
        super(ResizeGatedConv, self).__init__()
        
        self.gated_conv = GatedConv(in_channels, out_channels, kernel_size=3, stride=1)
        self.scale_factor = scale_factor
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor)
        return self.gated_conv(x)