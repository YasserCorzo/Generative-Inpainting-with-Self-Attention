import torch
import torch.nn as nn

from layers import GatedConv, ResizeGatedConv

class FreeFormImageInpaint(nn.module):
    def __init__(self, in_channels):
        super(FreeFormImageInpaint, self).__init__()
        
        self.coarse_network = nn.Sequential(
            GatedConv(in_channels, out_channels=32, kernel_size=5, stride=1), # batch_size x 32 x 256 x 256
            GatedConv(in_channels=32, out_channels=64, kernel_size=3, stride=2), # batch_size x 64 x 128 x 128
            GatedConv(in_channels=64, out_channels=64, kernel_size=3, stride=1), # batch_size x 64 x 128 x 128
            GatedConv(in_channels=64, out_channels=128, kernel_size=3, stride=2), # batch_size x 128 x 64 x 64
            GatedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1), # batch_size x 128 x 64 x 64
            GatedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1), # batch_size x 128 x 64 x 64
            GatedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=2), # batch_size x 128 x 64 x 64
            GatedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=4), # batch_size x 128 x 64 x 64
            GatedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=8), # batch_size x 128 x 64 x 64
            GatedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=16), # batch_size x 128 x 64 x 64
            GatedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1), # batch_size x 128 x 64 x 64
            GatedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1), # batch_size x 128 x 64 x 64
            ResizeGatedConv(in_channels=128, out_channels=64, scale_factor=2), # batch_size x 64 x 128 x 128
            GatedConv(in_channels=64, out_channels=64, kernel_size=3, stride=1), # batch_size x 64 x 128 x 128
            ResizeGatedConv(in_channels=64, out_channels=32), # batch_size x 32 x 256 x 256
            GatedConv(in_channels=32, out_channels=16, kernel_size=3, stride=1), # batch_size x 16 x 256 x 256
            GatedConv(in_channels=16, out_channels=3, kernel_size=3, stride=1, feature_act=None) # batch_size x 3 x 256 x 256                            
        )
    
    
    def forward(self, x, masks):
        """
        dim of x: batch_size x channels x 256 x 256
        """
        
        # TODO: normalize images and pair images with corresponding masks as input
        # input will contain masked images
        input = masked_images
        
        # coarse network
        coarse_out = self.coarse_network(input)
        # clip output so values are between -1 and 1
        coarse_clip = torch.clamp(coarse_out, -1, 1)

    def loss_function(self, x_hat, x, masks, alpha):
        # TODO: convert x/x_hat to just masked and unmasked portion
        masked, masked_hat, unmasked, unmasked_hat = 

        mask_bit_ratio = torch.mean(masks, -1) #take the ratio of masked to unmasked bits
        bit_mask_ratio = torch.mean(1 - masks, -1) #take the ratio of unmasked to masked bits
        masked_loss = alpha * torch.mean(torch.abs(masked - masked_hat) / mask_bit_ratio)
        unmasked_loss = alpha * torch.mean(torch.abs(unmasked - unmasked_hat) / mask_bit_ratio)
        loss = masked_loss + unmasked_loss
        return loss
