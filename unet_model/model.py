import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from dataset import UDataset
from torch.utils.data import DataLoader


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    def __init__(self, input_dim_AGRI=3, input_dim_GIIRS=3, output_dim=1):
        super(U_Net, self).__init__()

        '''
        encoder
        '''
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool_3 = nn.MaxPool2d(kernel_size=3, stride=3)

        # AGRI_curr+AGRI_prev分支
        self.Conv0_AGRI = conv_block(ch_in=input_dim_AGRI * 2, ch_out=32)
        self.Conv1_AGRI = conv_block(ch_in=32, ch_out=64)
        self.Conv2_AGRI = conv_block(ch_in=64, ch_out=128)
        self.Conv3_AGRI = conv_block(ch_in=128, ch_out=256)
        self.Conv4_AGRI = conv_block(ch_in=256, ch_out=512)
        self.Conv5_AGRI = conv_block(ch_in=512, ch_out=1024)

        # GIIRS分支
        self.Conv1_GIIRS = conv_block(ch_in=input_dim_GIIRS, ch_out=64)
        self.Conv2_GIIRS = conv_block(ch_in=64, ch_out=128)
        self.Conv3_GIIRS = conv_block(ch_in=128, ch_out=256)
        self.Conv4_GIIRS = conv_block(ch_in=256, ch_out=512)
        self.Conv5_GIIRS = conv_block(ch_in=512, ch_out=1024)

        # GIIRS_delta_time分支
        self.Conv1_time = conv_block(ch_in=1, ch_out=16)
        self.Conv2_time = conv_block(ch_in=16, ch_out=32)
        self.Conv3_time = conv_block(ch_in=32, ch_out=64)
        self.Conv4_time = conv_block(ch_in=64, ch_out=128)
        self.Conv5_time = conv_block(ch_in=128, ch_out=256)

        '''
        decoder
        '''
        # u分支
        self.Up5_u = up_conv(ch_in=2304, ch_out=512)
        self.Up_conv5_u = conv_block(ch_in=512 * 3, ch_out=512)

        self.Up4_u = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4_u = conv_block(ch_in=256 * 3, ch_out=256)

        self.Up3_u = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3_u = conv_block(ch_in=128 * 3, ch_out=128)

        self.Up2_u = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2_u = conv_block(ch_in=64 * 3, ch_out=64)

        self.Conv_1x1_u = nn.Conv2d(64, output_dim, kernel_size=1, stride=1, padding=0)

        # v分支
        self.Up5_v = up_conv(ch_in=2304, ch_out=512)
        self.Up_conv5_v = conv_block(ch_in=512 * 3, ch_out=512)

        self.Up4_v = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4_v = conv_block(ch_in=256 * 3, ch_out=256)

        self.Up3_v = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3_v = conv_block(ch_in=128 * 3, ch_out=128)

        self.Up2_v = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2_v = conv_block(ch_in=64 * 3, ch_out=64)

        self.Conv_1x1_v = nn.Conv2d(64, output_dim, kernel_size=1, stride=1, padding=0)

    def pad_to_four(self, x):
        h, w = x.shape[-2:]
        pad_h = h % 4
        pad_w = w % 4
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        return x

    def forward(self, AGRI_curr, AGRI_prev, GIIRS, GIIRS_delta_time):
        '''

        :param AGRI_curr: (b,15, 576, 648)
        :param AGRI_prev: (b,15, 576, 648)
        :param GIIRS: (b,1690, 192, 216)
        :param GIIRS_delta_time: (b, 192, 216)

        :return: d1_u (b,37,80,80), d1_v (b,37,80,80)
        '''

        '''
        encoding
        '''
        # AGRI_curr+AGRI_prev分支

        x0_AGRI = self.Conv0_AGRI(torch.cat([AGRI_curr, AGRI_prev], dim=1))  # (b,32,576,648)
        x1_AGRI = self.Maxpool_3(x0_AGRI)  # (b,32,192,216)
        x1_AGRI = self.Conv1_AGRI(x1_AGRI)  # (b,64,192,216)

        x2_AGRI = self.Maxpool(x1_AGRI)  # (b,64,96,108)
        x2_AGRI = self.Conv2_AGRI(x2_AGRI)  # (b,128,96,108)

        x3_AGRI = self.Maxpool(x2_AGRI)  # (b,128,48,54)
        x3_AGRI = self.Conv3_AGRI(x3_AGRI)  # (b,256,48,54)

        x3_AGRI = self.pad_to_four(x3_AGRI) # (b,256,48,56)
        x4_AGRI = self.Maxpool(x3_AGRI)  # (b,256,24,28)
        x4_AGRI = self.Conv4_AGRI(x4_AGRI)  # (b,512,24,28)

        x5_AGRI = self.Maxpool(x4_AGRI)  # (b,512,12,14)
        x5_AGRI = self.Conv5_AGRI(x5_AGRI)  # (b,1024,12,14)

        # GIIRS分支
        x1_GIIRS = self.Conv1_GIIRS(GIIRS)  # (b,64,192,216)

        x2_GIIRS = self.Maxpool(x1_GIIRS)  # (b,64,96,108)
        x2_GIIRS = self.Conv2_GIIRS(x2_GIIRS)  # (b,128,96,108)

        x3_GIIRS = self.Maxpool(x2_GIIRS)  # (b,128,48,54)
        x3_GIIRS = self.Conv3_GIIRS(x3_GIIRS)  # (b,256,48,54)

        x3_GIIRS = self.pad_to_four(x3_GIIRS) # (b,256,48,56)
        x4_GIIRS = self.Maxpool(x3_GIIRS)  # (b,256,24,28)
        x4_GIIRS = self.Conv4_GIIRS(x4_GIIRS)  # (b,512,24,28)

        x5_GIIRS = self.Maxpool(x4_GIIRS)  # (b,512,12,14)
        x5_GIIRS = self.Conv5_GIIRS(x5_GIIRS)  # (b,1024,12,14)

        # GIIRS_delta_time分支
        x1_time = self.Conv1_time(GIIRS_delta_time)  # (b,16,192,216)

        x2_time = self.Maxpool(x1_time)  # (b,16,96,108)
        x2_time = self.Conv2_time(x2_time)  # (b,32,96,108)

        x3_time = self.Maxpool(x2_time)  # (b,32,48,54)
        x3_time = self.Conv3_time(x3_time)  # (b,64,48,54)

        x3_time = self.pad_to_four(x3_time) # (b,64,48,56)
        x4_time = self.Maxpool(x3_time)  # (b,64,24,28)
        x4_time = self.Conv4_time(x4_time)  # (b,128,24,28)

        x5_time = self.Maxpool(x4_time)  # (b,128,12,14)
        x5_time = self.Conv5_time(x5_time)  # (b,256,12,14)

        '''
        AGRI+GIIRS - u decoding
        '''
        d5_u = self.Up5_u(torch.cat([x5_AGRI, x5_GIIRS,x5_time], dim=1))  # (b,512,24,28)
        d5_u = torch.cat((d5_u, x4_AGRI, x4_GIIRS), dim=1)  # (b,512*3,24,28)
        d5_u = self.Up_conv5_u(d5_u)  # (b,512,24,28)

        d4_u = self.Up4_u(d5_u)  # (b,256,48,56)
        d4_u = torch.cat((d4_u, x3_AGRI, x3_GIIRS), dim=1)  # (b,256*3,48,56)
        d4_u = self.Up_conv4_u(d4_u)  # (b,256,48,56)

        d4_u = d4_u[:,:,:,1:-1]

        d3_u = self.Up3_u(d4_u)  # (b,128,96,108)
        d3_u = torch.cat((d3_u, x2_AGRI, x2_GIIRS), dim=1)  # (b,128*3,96,108)
        d3_u = self.Up_conv3_u(d3_u)  # (b,128,96,108)

        d2_u = self.Up2_u(d3_u)  # (b,64,192,216)
        d2_u = torch.cat((d2_u, x1_AGRI, x1_GIIRS), dim=1)  # (b,64*3,192,216)
        d2_u = self.Up_conv2_u(d2_u)  # (b,64,192,216)

        d1_u = self.Conv_1x1_u(d2_u)  # (b,37,192,216)

        '''
        AGRI+GIIRS - v decoding
        '''
        d5_v = self.Up5_v(torch.cat([x5_AGRI, x5_GIIRS,x5_time], dim=1))  # (b,512,24,28)
        d5_v = torch.cat((d5_v, x4_AGRI, x4_GIIRS), dim=1)  # (b,512*3,24,28)
        d5_v = self.Up_conv5_v(d5_v)  # (b,512,24,28)

        d4_v = self.Up4_v(d5_v)  # (b,256,48,56)
        d4_v = torch.cat((d4_v, x3_AGRI, x3_GIIRS), dim=1)  # (b,256*3,48,56)
        d4_v = self.Up_conv4_v(d4_v)  # (b,256,48,56)

        d4_v = d4_v[:,:,:,1:-1]

        d3_v = self.Up3_v(d4_v)  # (b,128,96,108)
        d3_v = torch.cat((d3_v, x2_AGRI, x2_GIIRS), dim=1)  # (b,128*3,96,108)
        d3_v = self.Up_conv3_v(d3_v)  # (b,128,96,108)

        d2_v = self.Up2_v(d3_v)  # (b,64,192,216)
        d2_v = torch.cat((d2_v, x1_AGRI, x1_GIIRS), dim=1)  # (b,64*3,192,216)
        d2_v = self.Up_conv2_v(d2_v)  # (b,64,192,216)

        d1_v = self.Conv_1x1_v(d2_v)  # (b,37,192,216)

        return d1_u, d1_v


if __name__ == '__main__':
    input_dim_AGRI = 15
    input_dim_GIIRS = 1690
    output_dim = 37
    batch_size = 2
    # 初始化网络
    model = U_Net(input_dim_AGRI, input_dim_GIIRS, output_dim)
    AGRI_curr_n = torch.randn(batch_size, 15, 576, 648)
    AGRI_prev_n = torch.randn(batch_size, 15, 576, 648)
    GIIRS_n = torch.randn(batch_size, 1690, 192, 216)
    GIIRS_delta_time_n = torch.randn(batch_size,1, 192, 216)
    ERA5_u_n = torch.randn(batch_size, 37, 192, 216)
    ERA5_v_n = torch.randn(batch_size, 37, 192, 216)
    pred_u, pred_v = model(AGRI_curr_n, AGRI_prev_n, GIIRS_n, GIIRS_delta_time_n)
