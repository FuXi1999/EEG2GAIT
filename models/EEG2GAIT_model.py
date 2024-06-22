import torch
import torch.nn as nn

from utils.myaml import load_config
import torch.nn.functional as F
import time
import math



class EEG2GAIT(nn.Module):
    """
    test version.  ///v///
    """

    def __init__(self, config):
        super(EEG2GAIT, self).__init__()
        self.ts = config.eegnet.eeg.time_step
        self.config = config.eegnet
        self.drop_out = self.config.dropout
        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((self.config.blk1_kernel//2-1, self.config.blk1_kernel//2, 0, 0)),
            nn.Conv2d(
                in_channels=1,  # input shape (1, C, T)
                out_channels=self.config.F1,  # num_filters
                kernel_size=(1, self.config.blk1_kernel),  # filter size
                bias=False
            ),  # output shape (8, C, T)
            nn.BatchNorm2d(self.config.F1)  # output shape (8, C, T)
        )

       
        self.block_2 = nn.Sequential(

            nn.Conv2d(
                in_channels=self.config.F1,  # input shape (8, C, T)
                out_channels=self.config.D * self.config.F1,  # num_filters
                kernel_size=(self.config.num_chan_eeg, 1),  # filter size
                groups=self.config.F1,
                bias=False
            ),  # output shape (16, 1, T)
            nn.BatchNorm2d(self.config.D * self.config.F1),  # output shape (16, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (16, 1, T//4)
            nn.Dropout(self.drop_out)  # output shape (16, 1, T//4)
        )

        self.temporal_atten = nn.MultiheadAttention(self.config.F2, self.config.F2)
        
        self.mask = self.config.mask
        self.block_5 = nn.Sequential(
            nn.ZeroPad2d(((self.config.blk5_kernel + 1) // 2-1, (self.config.blk5_kernel + 1) // 2, 0, 0)),
            nn.Conv2d(
                in_channels=1,  # input shape (1, 32, T//4)
                out_channels=self.config.F2 * 2,  # num_filters
                kernel_size=(self.config.F2 * 2, self.config.blk5_kernel),  # filter size
                bias=False
            ),  # output shape (32, 1, T//4)

            nn.BatchNorm2d(self.config.F2 * 2),  # output shape (32, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 8)),# output shape (32, 1, T//32)
            nn.Dropout(self.drop_out)
        )
        self.out = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # input shape (1, 32, T//4)
                out_channels=self.config.num_chan_kin,  # num_filters
                kernel_size=(self.config.F2 * 2, int((self.config.eeg.time_step - self.mask) / 32)),
                bias=False
            ),  # output shape (32, 1, T//4)
        )
        self.flag = 0

    def forward(self, x):
        """

        Args:
            x: (Batch_size, 1, C, Tap_size)

        Returns:

        """
        if self.mask != 0:
            x = x[:,:,:,:-1 * self.mask]
        x = self.block_1(x)
        self.out1 = x
        self.out1.retain_grad()
        x = self.block_2(x)  # output shape (16, 1, T//4)
        x = torch.squeeze(x, 2)  # output shape (16, T//4)
        tmp_x = x
        x = x.permute(0, 2, 1)  # output shape (T//4, 16)
        x_tempo, _ = self.temporal_atten(x, x, x)  # output shape (T//4, 16)
        x_tempo = x_tempo.permute(0, 2, 1)  # output shape (F2, T//4)
        x = torch.cat((tmp_x, x_tempo), dim=1)  # output shape (F2 * 2, T//4)
        x = torch.unsqueeze(x, 1)# output shape (1, F2 * 2, T//4)
        x = self.block_5(x)
        self.feature = x
        x = x.permute(0,2,1,3)
        x = self.out(x)
        x = x.squeeze()
        return x




if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    x = torch.rand(100, 1, 59, 180).to(device)

    config = load_config('../config.yaml')
    model = EEG2GAIT(config).to(device)
    out = model(x)
    a = 0
