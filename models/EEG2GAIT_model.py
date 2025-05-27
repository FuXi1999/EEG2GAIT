import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d


class EEG2Gait(nn.Module):

    def __init__(self, config):
        super().__init__()
        nChan = config.num_chan_eeg
        nTime = config.eegnet.eeg.time_step
        pool_width = 3
        poolSize = {
                "LocalLayers": [(1, pool_width), (1, pool_width), (1, pool_width)],
                "GlobalLayers": (1, pool_width),
            }
        kernel_width = 10
        self.kernel_width = kernel_width
        localKernalSize = {
                "LocalLayers": [(1, kernel_width), (1, kernel_width), (1, kernel_width)],
                "GlobalLayers": (1, kernel_width),
            }
        nClass = config.eegnet.num_chan_kin
        dropoutP = 0.5
        nFilt_FirstLayer = 25
        nFiltLaterLayer = [25, 50, 100, 200]

        self.firstLayer = self.firstBlock(
            nFilt_FirstLayer,
            dropoutP,
            localKernalSize["LocalLayers"][0],
            nChan,
            poolSize["LocalLayers"][0],
        )
        # middleLayers = nn.Sequential(*[self.convBlock(inF, outF, dropoutP, localKernalSize)
        #     for inF, outF in zip(nFiltLaterLayer[:-1], nFiltLaterLayer[1:-1])])
        self.middleLayers = nn.Sequential(
            *[
                self.convBlock(inF, outF, dropoutP, kernalS, poolS)
                for inF, outF, kernalS, poolS in zip(
                    nFiltLaterLayer[:-2],
                    nFiltLaterLayer[1:-2],
                    localKernalSize["LocalLayers"][1:],
                    poolSize["LocalLayers"][1:],
                )
            ]
        )
        self.deform_conv_width = localKernalSize["LocalLayers"][-3][1]
        self.deform_conv = DeformConv2d(
            in_channels=nFiltLaterLayer[-3],
            out_channels=nFiltLaterLayer[-2],
            kernel_size=(1, self.deform_conv_width),
            padding=((self.deform_conv_width-1) // 2, self.deform_conv_width // 2)
        )
        
        self.firstGlobalLayer = self.convBlock(
            nFiltLaterLayer[-2],
            nFiltLaterLayer[-1],
            dropoutP,
            localKernalSize["GlobalLayers"],
            poolSize["GlobalLayers"],
        )

        self.allButLastLayers = nn.Sequential(
            self.firstLayer, self.middleLayers, self.firstGlobalLayer
        )

        self.lastLayer = self.lastBlock(nFiltLaterLayer[-1], nClass, (1, 2))

        
        self.weight_keys = [
            [
                "allButLastLayers.0.0.weight",
                "allButLastLayers.0.0.bias",
                "allButLastLayers.0.1.weight",
            ],
            ["allButLastLayers.1.0.1.weight"],
            ["allButLastLayers.1.1.1.weight"],
            ["allButLastLayers.2.1.weight"],
            ["lastLayer.0.weight", "lastLayer.0.bias"],
        ]

        # 多尺度卷积分支
        self.multi_scale_conv = nn.ModuleList([
            nn.Sequential(
                nn.ZeroPad2d(((ks-1) // 2, ks // 2, 0, 0)),
                nn.Conv2d(1, 1, kernel_size=(1, ks)),
                nn.ELU(),
            )
            for ks in [5, 10, 20]  # 短、中、长三个尺度
        ])

        # 相位门控机制
        self.phase_gate = nn.Sequential(
            nn.Conv2d(nFiltLaterLayer[-1], nFiltLaterLayer[-1], kernel_size=(1, 1)),
            nn.Sigmoid()
        )

    def convBlock(self, inF, outF, dropoutP, kernalSize, poolSize, *args, **kwargs):
        return nn.Sequential(
            nn.Dropout(p=dropoutP),
            nn.ZeroPad2d((kernalSize[1] // 2 - 1, kernalSize[1] // 2, 0, 0)),
            Conv2dWithConstraint(
                inF, outF, kernalSize, bias=False, max_norm=2, *args, **kwargs
            ),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.MaxPool2d(poolSize, stride=poolSize),
        )

    def firstBlock(self, outF, dropoutP, kernalSize, nChan, poolSize, *args, **kwargs):
        return nn.Sequential(

            nn.ZeroPad2d((kernalSize[1] // 2 - 1, kernalSize[1] // 2, 0, 0)),
            Conv2dWithConstraint(
                3, outF, kernalSize, padding=0, max_norm=2, *args, **kwargs
            ),
            Conv2dWithConstraint(25, 25, (nChan, 1), padding=0, bias=False, max_norm=2),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.MaxPool2d(poolSize, stride=poolSize),
        )

    def lastBlock(self, inF, outF, kernalSize, *args, **kwargs):
        return nn.Sequential(

            # nn.ZeroPad2d((kernalSize[1] // 2 - 1, kernalSize[1] // 2, 0, 0)),
            Conv2dWithConstraint(inF, outF, kernalSize, max_norm=0.5, *args, **kwargs),
            # nn.LogSoftmax(dim=1),
        )

    def calculateOutSize(self, model, nChan, nTime):
        """
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        """
        data = torch.rand(1, 1, nChan, nTime)
        model.eval()
        out = model(data).shape
        return out[2:]

    def forward(self, x):
        multi_scale_features = [conv(x) for conv in self.multi_scale_conv]
        x = torch.cat(multi_scale_features, dim=1) 

        
        x = self.firstLayer(x)
        x = self.middleLayers(x)
        offset = torch.zeros((x.size(0), 2 * self.deform_conv_width, x.size(2), x.size(3)))  
        x = self.deform_conv(x, offset)
        x = self.firstGlobalLayer(x)

        
        phase_weight = self.phase_gate(x)
        x = x * phase_weight  

        x = self.lastLayer(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)

        return x


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super().forward(x)
