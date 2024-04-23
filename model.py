import torch
import torch.nn as nn

from layers import GatedConv, ResizeGatedConv

class FreeFormImageInpaint(nn.Module):
    def __init__(self, in_channels):
        super(FreeFormImageInpaint, self).__init__()

        self.coarse_network = nn.Sequential(
            GatedConv(in_channels, out_channels=32, kernel_size=5, stride=1, padding=2), # batch_size x 32 x 256 x 256
            GatedConv(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), # batch_size x 64 x 128 x 128
            GatedConv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # batch_size x 64 x 128 x 128
            GatedConv(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1), # batch_size x 128 x 64 x 64
            GatedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # batch_size x 128 x 64 x 64
            GatedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # batch_size x 128 x 64 x 64
            GatedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2), # batch_size x 128 x 64 x 64
            GatedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4), # batch_size x 128 x 64 x 64
            GatedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=8, dilation=8), # batch_size x 128 x 64 x 64
            GatedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=16, dilation=16), # batch_size x 128 x 64 x 64
            GatedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # batch_size x 128 x 64 x 64
            GatedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # batch_size x 128 x 64 x 64
            ResizeGatedConv(in_channels=128, out_channels=64, padding=1, scale_factor=2), # batch_size x 64 x 128 x 128
            GatedConv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # batch_size x 64 x 128 x 128
            ResizeGatedConv(in_channels=64, out_channels=32, padding=1, scale_factor=2), # batch_size x 32 x 256 x 256
            GatedConv(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1), # batch_size x 16 x 256 x 256
            GatedConv(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1, feature_act=None) # batch_size x 3 x 256 x 256
        )
    
    
    def forward(self, x, masks):
        """
        dim of x: batch_size x 256 x 256 x channels
        dim of mask: batch_size x 256 x 256
        """
        #print("shape of images (B x C x H x W):", x.shape)
        #print("shape of masks (B x H x W):", masks.shape)
        # TODO: normalize images and pair images with corresponding masks as input
        # input will contain masked images
        #x = x.permute(0, 3, 1, 2) # batch_size x channels x 256 x 256
        masks = masks.unsqueeze(1) # batch_size x 1 x 256 x 256
        masked_imgs = x * (1 - masks)
        #print(masks.shape)
        #print(masked_imgs.shape)
        input = torch.cat([masked_imgs, masks], dim=1) # batch_size x (channels + 1) x 256 x 256
        #print("shape of input into coarse network:", input.shape)

        # coarse network
        coarse_out = self.coarse_network(input)
        # clip output so values are between -1 and 1
        coarse_clip = torch.clamp(coarse_out, -1, 1)

        return coarse_clip
    
    def loss_function(self, x_hat, x, masks, alpha):
        '''
        dim of x_hat & x: batch_size x 3 x 256 x 256
        dim of masks: batch_size x 256 x 256
        '''

        # TODO: convert x/x_hat to just masked and unmasked portion
        #print("shape of x:", x.shape)
        #print("shape of xhat:", x_hat.shape)
        masks = masks.unsqueeze(1) # batch_size x 1 x 256 x 256
        #print("shape of masks:", masks.shape)
        unmasked = x * masks
        unmasked_hat = x_hat * masks
        masked = x * (1 - masks)
        masked_hat = x_hat * (1 - masks)

        mask_bit_ratio = torch.mean(torch.mean(masks, -1), -1) #take the ratio of masked to unmasked bits
        mask_bit_ratio = mask_bit_ratio.unsqueeze(-1)
        mask_bit_ratio = mask_bit_ratio.unsqueeze(-1)
        #print(mask_bit_ratio.shape)
        bit_mask_ratio = torch.mean(torch.mean(1-masks, -1), -1) #take the ratio of unmasked to masked bits
        bit_mask_ratio = bit_mask_ratio.unsqueeze(-1)
        bit_mask_ratio = bit_mask_ratio.unsqueeze(-1)
        masked_loss = alpha * torch.mean(torch.abs(masked - masked_hat) / mask_bit_ratio)
        unmasked_loss = alpha * torch.mean(torch.abs(unmasked - unmasked_hat) / bit_mask_ratio)
        loss = masked_loss + unmasked_loss
        return loss
