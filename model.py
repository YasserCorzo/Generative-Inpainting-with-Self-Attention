import torch
import torch.nn as nn

from layers import GatedConv, ResizeGatedConv, SpectralNormConv


def normalize_tensor(data: torch.Tensor,
                     smin: float,
                     smax: float,
                     tmin : float,
                     tmax : float) -> torch.Tensor:

    slength = smax - smin
    tlength = tmax - tmin
    data = (data - smin) / slength
    data = (data * tlength) + tmin
    return data

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            SpectralNormConv(in_channels=4, out_channels=64, kernel_size=5, stride=2, padding=2), # batch_size x 4 x 256 x 256 ---> batch_size x 64 x 128 x 128
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
        #x = normalize_tensor(x, 0, 255, -1, 1)
        masks = masks.unsqueeze(1) # batch_size x 1 x 256 x 256
        input = torch.cat([x, masks], dim=1) # batch_size x (channels + 1) x 256 x 256
        out = self.layers(input) 
        return out
    
    #Discriminator Loss
    def loss_function(self, x, x_hat):
        '''
        dim of x_hat & x: batch_size x 3 x 256 x 256
        '''
        return torch.mean(nn.functional.relu(1 - x)) + torch.mean(nn.functional.relu(1 + x_hat))
                        
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.coarse_network = nn.Sequential(
            GatedConv(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2), # batch_size x 32 x 256 x 256
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
            GatedConv(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2), # batch_size x 32 x 256 x 256
            GatedConv(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1), # batch_size x 32 x 128 x 128
            GatedConv(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # batch_size x 64 x 128 x 128
            GatedConv(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1), # batch_size x 128 x 64 x 64
            GatedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # batch_size x 128 x 64 x 64
            GatedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # batch_size x 128 x 64 x 64

            SelfAttention(in_dim=128),                                                        # batch_size x 128 x 64 x 64

            GatedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # batch_size x 128 x 64 x 64
            GatedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # batch_size x 128 x 64 x 64

            GatedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # batch_size x 128 x 64 x 64
            GatedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # batch_size x 128 x 64 x 64
            ResizeGatedConv(in_channels=128, out_channels=64, stride=1, padding=1), # batch_size x 64 x 128 x 128
            GatedConv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # batch_size x 64 x 128 x 128
            ResizeGatedConv(in_channels=64, out_channels=32, stride=1, padding=1), # batch_size x 32 x 256 x 256
            GatedConv(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1), # batch_size x 16 x 256 x 256
            GatedConv(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1, feature_act=None)  # batch_size x 3 x 256 x 256
        )
    
    def forward(self, x, masks):
        """
        dim of x: batch_size x channels x H x W
        dim of mask: batch_size x H x W
        """
        masks = masks.unsqueeze(1) # batch_size x 1 x 256 x 256
        masked_imgs = x * (1 - masks)

        input = torch.cat([masked_imgs, masks], dim=1) # batch_size x (channels + 1) x 256 x 256

        # coarse network
        coarse_out = self.coarse_network(input)
        # clip output so values are between -1 and 1
        coarse_clip = torch.clamp(coarse_out, -1.0, 1.0)

        coarse_raw = coarse_clip

        # process coarse network output for refinement network input
        coarse_processed = coarse_clip * masks + masked_imgs

        # refinement network
        refine_in = torch.cat([coarse_processed, masks], dim=1)
        refine_out = self.refinement_network(refine_in)
        refine_clip = torch.clamp(refine_out, -1.0, 1.0)

        refine_raw = refine_clip

        # merge original image with refinement
        reconstructed_image = refine_clip * masks + x * (1 - masks)

        return reconstructed_image, coarse_raw, refine_raw
        
    def recon_loss_function(self, x_hat, x, masks, alpha):
        '''
        dim of x_hat & x: batch_size x 3 x H x W
        dim of masks: batch_size x H x W
        '''

        # TODO: convert x/x_hat to just masked and unmasked portion
        masks = masks.unsqueeze(1) # batch_size x 1 x 256 x 256
        unmasked = x * masks
        unmasked_hat = x_hat * masks
        masked = x * (1 - masks)
        masked_hat = x_hat * (1 - masks)

        mask_bit_ratio = torch.mean(torch.mean(masks, -1), -1) #take the ratio of masked to unmasked bits
        mask_bit_ratio = mask_bit_ratio.unsqueeze(-1)
        mask_bit_ratio = mask_bit_ratio.unsqueeze(-1)

        bit_mask_ratio = torch.mean(torch.mean(1-masks, -1), -1) #take the ratio of unmasked to masked bits
        bit_mask_ratio = bit_mask_ratio.unsqueeze(-1)
        bit_mask_ratio = bit_mask_ratio.unsqueeze(-1)
        masked_loss = alpha * torch.mean(torch.abs(masked - masked_hat) / mask_bit_ratio)
        unmasked_loss = alpha * torch.mean(torch.abs(unmasked - unmasked_hat) / bit_mask_ratio)
        loss = masked_loss + unmasked_loss
        return loss

    def combined_rec_loss_function(self, x, x_coarse, x_refinement, masks, alpha):
        '''
        dim of x & x_coarse & x_refinement: batch_size x channel x H x W
        dim of masks: batch_size x H x W 
        '''
        coarse_recon = self.recon_loss_function(x_coarse, x, masks, alpha)
        refinement_recon = self.recon_loss_function(x_refinement, x, masks, alpha)

        return coarse_recon + refinement_recon
    
    def loss_function(self, x_hat):
        '''
        dim of x_hat & x: batch_size x 3 x H x W
        '''
        return (-1) * torch.mean(x_hat)

class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim=128,activation='relu',with_attn=False):
        super(SelfAttention,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.with_attn = with_attn
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        if self.with_attn:
            return out, attention
        else:
            return out

class InpaintModel(nn.Module):
    def __init__(self):
        super(InpaintModel, self).__init__()

        self.generator = Generator()
        self.discriminator = Discriminator()

    def forward(self, images, masks):
        """
        dim of images: batch_size x channels x H x W
        dim of mask: batch_size x H x W
        """
        gen_images = self.generator(images, masks)
        return gen_images
