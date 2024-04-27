import torch
import torch.nn as nn

from layers import GatedConv, ResizeGatedConv, SpectralNormConv, Convolution


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            SpectralNormConv(in_channels, out_channels=64, kernel_size=5, stride=2, padding=2), # batch_size x 4 x 256 x 256 ---> batch_size x 64 x 128 x 128
            SpectralNormConv(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2), # batch_size x 128 x 64 x 64
            SpectralNormConv(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2), # batch_size x 256 x 32 x 32
            SpectralNormConv(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=2), # batch_size x 256 x 16 x 16
            SpectralNormConv(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=2), # batch_size x 256 x 8 x 8
            SpectralNormConv(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=2), # batch_size x 256 x 4 x 4
            nn.Flatten() # 1 x 4096
        )

    def forward(self, x, masks):
        """
        dim of x: batch_size x channels x H x W
        dim of masks: batch_size x H x W
        """
        masks = masks.unsqueeze(1) # batch_size x 1 x 256 x 256
        input = torch.cat([x, masks], dim=1) # batch_size x (channels + 1) x 256 x 256
        out = self.layers(input) 
        return out
    
    #Discriminator Loss
    def loss_function(self, x_hat, x):
        '''
        dim of x_hat & x: batch_size x 3 x 256 x 256
        '''
        return torch.mean(nn.functional.relu(1 - x)) + torch.mean(nn.functional.relu(1 + x_hat))
        
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

        self.refinement_network = nn.Sequential(
            GatedConv(in_channels, out_channels=32, kernel_size=5, stride=1, padding=2), 
            GatedConv(in_channels=2, out_channels=32, kernel_size=3, stride=2, padding=1), 
            GatedConv(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), 
            GatedConv(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1), 
            GatedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), 
            GatedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), 
            SelfAttention(in_channels=128), 
            GatedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), 
            GatedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), 
            GatedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), 
            GatedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), 
            ResizeGatedConv(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1), 
            GatedConv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), 
            ResizeGatedConv(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1), 
            GatedConv(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1), 
            GatedConv(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1, feature_act=None)
        )

        self.discrimininator = Discriminator()
    
    def forward(self, x, masks):
        """
        dim of x: batch_size x channels x H x W
        dim of mask: batch_size x H x W
        """
        #print("shape of images (B x C x H x W):", x.shape)
        #print("shape of masks (B x H x W):", masks.shape)
        # TODO: normalize images and pair images with corresponding masks as input
        # input will contain masked images
        #x = x.permute(0, 3, 1, 2) # batch_size x channels x 256 x 256
        masks = masks.unsqueeze(1) # batch_size x 1 x 256 x 256
        masked_imgs = (x * (1 - masks)) + masks
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
        dim of x_hat & x: batch_size x 3 x H x W
        dim of masks: batch_size x H x W
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

class SelfAttention(nn.Module):

    def __init__(self, in_channels: int = 3,
                       inter_channels: int = None):

        super(SelfAttention, self).__init__()
        if inter_channels is None:
            inter_channels = in_channels // 8
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.gamma = nn.Parameter(torch.zeros(1))

        self.conv_key = nn.Conv2d(in_channels=in_channels,
                                  out_channels=inter_channels,
                                  kernel_size=1)

        self.conv_query = nn.Conv2d(in_channels=in_channels,
                                    out_channels=inter_channels,
                                    kernel_size=1)

        self.conv_value = nn.Conv2d(in_channels=in_channels,
                                    out_channels=inter_channels,
                                    kernel_size=1)

        self.conv_final = nn.Conv2d(in_channels=inter_channels,
                                    out_channels=in_channels,
                                    kernel_size=1)

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        channels = x.shape[1]
        height = x.shape[2]
        width = x.shape[3]

        #key, query, value compute
        key = self.conv_key(x)
        query = self.conv_query(x)
        value = self.conv_value(x)

        #resizing
        key = key.view(batch_size, self.inter_channels, height * width)
        query = query.view(batch_size, self.inter_channels, height * width)
        value = value.view(batch_size, self.inter_channels, height * width)
        query = query.permute(0, 2, 1)

        #get attention scores
        attention = torch.bmm(query, key)
        #normalize attention score with softmax
        attention = torch.softmax(attention, dim=1)
        
        #multiplying attention and value to get output
        attention_value = torch.bmm(value, attention)
        attention_value = attention_value.view(batch_size, self.inter_channels, height, width)

        #final conv layer
        output = self.conv_final(attention_value)
        output = self.gamma * output + x

        return output, attention

##### For Generator Loss
    #Generator Loss
    def loss_function(self, x_hat):
        '''
        dim of x_hat: batch_size x 3 x 256 x 256
        '''
        return -1 * torch.mean(x_hat)
