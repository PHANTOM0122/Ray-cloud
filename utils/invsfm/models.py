import torch
import torch.nn as nn

#-----------------------------
# batch normalization and activation function after convolution layer in encoder
class VisibNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        if in_channels < 5:
        # SIFT descriptor X
            self.ech = [64,128,256,512,512,512]
        else:
        # SIFT descriptor O
            self.ech = [256,256,256,512,512,512]
        self.dch = [512,512,512,256,256,256,128,64,32,1]

        # Encoder
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, self.ech[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.ech[0], eps=1e-3, momentum=0., affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(self.ech[0], self.ech[1], kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.ech[1], eps=1e-3, momentum=0., affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(self.ech[1], self.ech[2], kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.ech[2], eps=1e-3, momentum=0., affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(self.ech[2], self.ech[3], kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.ech[3], eps=1e-3, momentum=0., affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(self.ech[3], self.ech[4], kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.ech[4], eps=1e-3, momentum=0., affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.down6 = nn.Sequential(
            nn.Conv2d(self.ech[4], self.ech[5], kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.ech[5], eps=1e-3, momentum=0., affine=True, track_running_stats=True),
            nn.ReLU()
        )

        # Decoder
        self.up1 = nn.Sequential(
            nn.Conv2d(self.ech[5], self.dch[0], kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.dch[0], eps=1e-3, momentum=0., affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(self.dch[0]+self.ech[4], self.dch[1], kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.dch[1], eps=1e-3, momentum=0., affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(self.dch[1]+self.ech[3], self.dch[2], kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.dch[2], eps=1e-3, momentum=0., affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(self.dch[2]+self.ech[2], self.dch[3], kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.dch[3], eps=1e-3, momentum=0., affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.up5 = nn.Sequential(
            nn.Conv2d(self.dch[3]+self.ech[1], self.dch[4], kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.dch[4], eps=1e-3, momentum=0., affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.up6 = nn.Sequential(
            nn.Conv2d(self.dch[4]+self.ech[0], self.dch[5], kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.dch[5], eps=1e-3, momentum=0., affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.up7 = nn.Sequential(
            nn.Conv2d(self.dch[5]+self.in_channels, self.dch[6], kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.dch[6], eps=1e-3, momentum=0., affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.up8 = nn.Sequential(
            nn.Conv2d(self.dch[6], self.dch[7], kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.dch[7], eps=1e-3, momentum=0., affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.up9 = nn.Sequential(
            nn.Conv2d(self.dch[7], self.dch[8], kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.dch[8], eps=1e-3, momentum=0., affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.up10 = nn.Sequential(
            nn.Conv2d(self.dch[8], self.dch[9], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.Sigmoid()
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')


    def forward(self, x):
        dx0 = x
        dx1 = self.down1(x)     # in_c  -> 256
        dx2 = self.down2(dx1)   # 256   -> 256
        dx3 = self.down3(dx2)   # 256   -> 256
        dx4 = self.down4(dx3)   # 256   -> 512
        dx5 = self.down5(dx4)   # 512   -> 512
        dx6 = self.down6(dx5)   # 512   -> 512

        # # upsampling -> Conv -> BatchNorm -> ReLU -> concat
        x = self.upsample(dx6)
        x = self.up1(x)         # 512 -> 512

        x = torch.cat([dx5,x], dim=1)   # 512 -> 512+512
        x = self.upsample(x)
        x = self.up2(x)         # 1024 -> 512

        x = torch.cat([dx4,x], dim=1)   # 512 -> 512+512
        x = self.upsample(x)
        x = self.up3(x)         # 1024 -> 512

        x = torch.cat([dx3,x], dim=1)   # 512 -> 256+512
        x = self.upsample(x)
        x = self.up4(x)         # 768 -> 256

        x = torch.cat([dx2,x], dim=1)   # 256 -> 256+256
        x = self.upsample(x)
        x = self.up5(x)         # 512 -> 256

        x = torch.cat([dx1,x], dim=1)   # 256 -> 256+256
        x = self.upsample(x)
        x = self.up6(x)         # 512 -> 256

        x = torch.cat([dx0,x], dim=1)   # 256 -> 132+256
        x = self.up7(x)         # 388 -> 128

        x = self.up8(x)         # 128 -> 64
        x = self.up9(x)         # 64 -> 32
        x = self.up10(x)        # 32 -> 1
        return x
         
class CoarseNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.ech = [256,256,256,512,512,512]
        self.dch = [512,512,512,256,256,256,128,64,32,3]

        # Encoder
        self.down1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.ech[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.ech[0], eps=1e-3, momentum=0., affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(self.ech[0], self.ech[1], kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.ech[1], eps=1e-3, momentum=0., affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(self.ech[1], self.ech[2], kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.ech[2], eps=1e-3, momentum=0., affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(self.ech[2], self.ech[3], kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.ech[3], eps=1e-3, momentum=0., affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(self.ech[3], self.ech[4], kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.ech[4], eps=1e-3, momentum=0., affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.down6 = nn.Sequential(
            nn.Conv2d(self.ech[4], self.ech[5], kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.ech[5], eps=1e-3, momentum=0., affine=True, track_running_stats=True),
            nn.ReLU()
        )

        # Decoder
        self.up1 = nn.Sequential(
            nn.Conv2d(self.ech[5], self.dch[0], kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.dch[0], eps=1e-3, momentum=0., affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(self.dch[0]+self.ech[4], self.dch[1], kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.dch[1], eps=1e-3, momentum=0., affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(self.dch[1]+self.ech[3], self.dch[2], kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.dch[2], eps=1e-3, momentum=0., affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(self.dch[2]+self.ech[2], self.dch[3], kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.dch[3], eps=1e-3, momentum=0., affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.up5 = nn.Sequential(
            nn.Conv2d(self.dch[3]+self.ech[1], self.dch[4], kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.dch[4], eps=1e-3, momentum=0., affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.up6 = nn.Sequential(
            nn.Conv2d(self.dch[4]+self.ech[0], self.dch[5], kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.dch[5], eps=1e-3, momentum=0., affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.up7 = nn.Sequential(
            nn.Conv2d(self.dch[5]+self.in_channels, self.dch[6], kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.dch[6], eps=1e-3, momentum=0., affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.up8 = nn.Sequential(
            nn.Conv2d(self.dch[6], self.dch[7], kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.dch[7], eps=1e-3, momentum=0., affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.up9 = nn.Sequential(
            nn.Conv2d(self.dch[7], self.dch[8], kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.dch[8], eps=1e-3, momentum=0., affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.up10 = nn.Sequential(
            nn.Conv2d(self.dch[8], self.dch[9], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.Tanh()
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, x):
        dx0 = x
        dx1 = self.down1(x)     # in_c  -> 256
        dx2 = self.down2(dx1)   # 256   -> 256
        dx3 = self.down3(dx2)   # 256   -> 256
        dx4 = self.down4(dx3)   # 256   -> 512
        dx5 = self.down5(dx4)   # 512   -> 512
        dx6 = self.down6(dx5)   # 512   -> 512

        # # upsampling -> Conv -> BatchNorm -> ReLU -> concat
        x = self.upsample(dx6)
        x = self.up1(x)         # 512 -> 512

        x = torch.cat([dx5,x], dim=1)   # 512 -> 512+512
        x = self.upsample(x)
        x = self.up2(x)         # 1024 -> 512

        x = torch.cat([dx4,x], dim=1)   # 512 -> 512+512
        x = self.upsample(x)
        x = self.up3(x)         # 1024 -> 512

        x = torch.cat([dx3,x], dim=1)   # 512 -> 256+512
        x = self.upsample(x)
        x = self.up4(x)         # 768 -> 256

        x = torch.cat([dx2,x], dim=1)   # 256 -> 256+256
        x = self.upsample(x)
        x = self.up5(x)         # 512 -> 256

        x = torch.cat([dx1,x], dim=1)   # 256 -> 256+256
        x = self.upsample(x)
        x = self.up6(x)         # 512 -> 256

        x = torch.cat([dx0,x], dim=1)   # 256 -> 132+256
        x = self.up7(x)         # 388 -> 128

        x = self.up8(x)         # 128 -> 64
        x = self.up9(x)         # 64 -> 32
        x = self.up10(x)        # 32 -> 1

        return x

class RefineNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.ech = [256,256,256,512,512,512]
        self.dch = [512,512,512,256,256,256,128,64,32,3]
        # Encoder
        self.down1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.ech[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.ech[0], eps=1e-3, momentum=None, affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(self.ech[0], self.ech[1], kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.ech[1], eps=1e-3, momentum=None, affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(self.ech[1], self.ech[2], kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.ech[2], eps=1e-3, momentum=None, affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(self.ech[2], self.ech[3], kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.ech[3], eps=1e-3, momentum=None, affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(self.ech[3], self.ech[4], kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.ech[4], eps=1e-3, momentum=None, affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.down6 = nn.Sequential(
            nn.Conv2d(self.ech[4], self.ech[5], kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.ech[5], eps=1e-3, momentum=None, affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # Decoder
        self.up1 = nn.Sequential(
            nn.Conv2d(self.ech[5], self.dch[0], kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.dch[0], eps=1e-3, momentum=None, affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(self.dch[0]+self.ech[4], self.dch[1], kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.dch[1], eps=1e-3, momentum=None, affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(self.dch[1]+self.ech[3], self.dch[2], kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.dch[2], eps=1e-3, momentum=None, affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(self.dch[2]+self.ech[2], self.dch[3], kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.dch[3], eps=1e-3, momentum=None, affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.up5 = nn.Sequential(
            nn.Conv2d(self.dch[3]+self.ech[1], self.dch[4], kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.dch[4], eps=1e-3, momentum=None, affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.up6 = nn.Sequential(
            nn.Conv2d(self.dch[4], self.dch[5], kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.dch[5], eps=1e-3, momentum=None, affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.up7 = nn.Sequential(
            nn.Conv2d(self.dch[5], self.dch[6], kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.dch[6], eps=1e-3, momentum=None, affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.up8 = nn.Sequential(
            nn.Conv2d(self.dch[6], self.dch[7], kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.dch[7], eps=1e-3, momentum=None, affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.up9 = nn.Sequential(
            nn.Conv2d(self.dch[7], self.dch[8], kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.dch[8], eps=1e-3, momentum=None, affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.up10 = nn.Sequential(
            nn.Conv2d(self.dch[8], self.dch[9], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.Tanh()
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        dx0 = x
        dx1 = self.down1(x)     # in_c  -> 256
        dx2 = self.down2(dx1)   # 256   -> 256
        dx3 = self.down3(dx2)   # 256   -> 256
        dx4 = self.down4(dx3)   # 256   -> 512
        dx5 = self.down5(dx4)   # 512   -> 512
        dx6 = self.down6(dx5)   # 512   -> 512

        # # upsampling -> Conv -> BatchNorm -> ReLU -> concat
        x = self.upsample(dx6)
        x = self.up1(x)         # 512 -> 512

        x = torch.cat([dx5,x], dim=1)   # 512 -> 512+512
        x = self.upsample(x)
        x = self.up2(x)         # 1024 -> 512

        x = torch.cat([dx4,x], dim=1)   # 512 -> 512+512
        x = self.upsample(x)
        x = self.up3(x)         # 1024 -> 512

        x = torch.cat([dx3,x], dim=1)   # 512 -> 256+512
        x = self.upsample(x)
        x = self.up4(x)         # 768 -> 256

        x = torch.cat([dx2,x], dim=1)   # 256 -> 256+256
        x = self.upsample(x)
        x = self.up5(x)         # 512 -> 256

        x = self.upsample(x)
        x = self.up6(x)         # 256 -> 256

        x = self.up7(x)         # 256 -> 128

        x = self.up8(x)         # 128 -> 64
        x = self.up9(x)         # 64 -> 32
        x = self.up10(x)        # 32 -> 1
        return x
