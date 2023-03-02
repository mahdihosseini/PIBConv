import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
    'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride),
                  padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1),
                  padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
    ),
'''

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv1_3x3': lambda C, stride, affine: SepConvInverted1 (C, C, 3, stride, 1, affine=affine),
    'sep_conv1_5x5': lambda C, stride, affine: SepConvInverted1 (C, C, 5, stride, 2, affine=affine),
    'sep_conv1_7x7': lambda C, stride, affine: SepConvInverted1 (C, C, 7, stride, 3, affine=affine),
    'sep_conv2_3x3': lambda C, stride, affine: SepConvInverted2 (C, C, 3, stride, 1, affine=affine),
    'sep_conv2_5x5': lambda C, stride, affine: SepConvInverted2 (C, C, 5, stride, 2, affine=affine),
    'sep_conv2_7x7': lambda C, stride, affine: SepConvInverted2 (C, C, 7, stride, 3, affine=affine),
    'sep_conv3_3x3': lambda C, stride, affine: SepConvInverted3 (C, C, 3, stride, 1, affine=affine),
    'sep_conv3_5x5': lambda C, stride, affine: SepConvInverted3 (C, C, 5, stride, 2, affine=affine),
    'sep_conv3_7x7': lambda C, stride, affine: SepConvInverted3 (C, C, 7, stride, 3, affine=affine),
    'sep_conv4_3x3': lambda C, stride, affine: SepConvInverted4 (C, C, 3, stride, 1, affine=affine),
    'sep_conv4_5x5': lambda C, stride, affine: SepConvInverted4 (C, C, 5, stride, 2, affine=affine),
    'sep_conv4_7x7': lambda C, stride, affine: SepConvInverted4 (C, C, 7, stride, 3, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv_new_conv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv_new_conv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv_new_conv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride),
                  padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1),
                  padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
    )
}


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first","channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
            

class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1,
                      padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class LNormReduce (nn.Module):

    def __init__(self, C_in):
        super(LNormReduce, self).__init__()

        self.op = nn.Sequential(
            LayerNorm(C_in,eps=1e-5,data_format="channels_first"),
            nn.Conv2d(C_in, C_in, kernel_size=2,stride=2, groups=C_in, bias=False)
        )

    def forward(self, x):
        return self.op(x)




class SepConvInverted1 (nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConvInverted1, self).__init__()

        layers = []

        if stride !=1: 
            layers.extend ([
                LayerNorm(C_in,eps=1e-5,data_format="channels_first"),
                nn.Conv2d(C_in, C_in, kernel_size=2,stride=2, bias=False)
            ])

        #print("==== sepconvinverted 1 ====")

        #print(f"{C_in}")
        layers.extend ([
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            LayerNorm(C_in,eps=1e-5,data_format="channels_first"),
            nn.Conv2d(C_in, C_in * 2, kernel_size=1, padding=0, bias=False),
            nn.GELU(),
            nn.Conv2d(C_in * 2, C_in, kernel_size=1,padding=0, bias=False),

            # == stacked ==
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            LayerNorm(C_in,eps=1e-5,data_format="channels_first"),
            nn.Conv2d(C_in, C_in * 2, kernel_size=1, padding=0, bias=False),
            nn.GELU(),
            nn.Conv2d(C_in * 2, C_out, kernel_size=1,padding=0, bias=False),
            ])
        print(f"{C_out}")
        self.op = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.op(x)


class SepConvInverted2 (nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConvInverted2, self).__init__()
 
        layers = []

        if stride !=1: 
            layers.extend ([
                LayerNorm(C_in,eps=1e-5,data_format="channels_first"),
                nn.Conv2d(C_in, C_in, kernel_size=2,stride=2, bias=False)
            ])

        layers.extend([
            LayerNorm(C_in,eps=1e-5,data_format="channels_first"),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in * 2, kernel_size=1, padding=0, bias=False),
            nn.GELU(),
            nn.Conv2d(C_in * 2, C_out, kernel_size=1,padding=0, bias=False),
            ])

        self.op = nn.Sequential(*layers)

    def forward(self, x):
        return self.op(x)

class SepConvInverted3 (nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConvInverted3, self).__init__()
        
        layers = []

        if stride !=1: 
            layers.extend ([
                LayerNorm(C_in,eps=1e-5,data_format="channels_first"),
                nn.Conv2d(C_in, C_in, kernel_size=2,stride=2, bias=False)
            ])

        layers.extend([
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in * 2, kernel_size=1, padding=0, bias=False),
            nn.GELU(),
            nn.Conv2d(C_in * 2, C_out, kernel_size=1,padding=0, bias=False),
            ])
        self.op = nn.Sequential(*layers)


    def forward(self, x):
        return self.op(x)

class SepConvInverted4 (nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConvInverted4, self).__init__()

        layers = []

        if stride !=1: 
            layers.extend ([
                LayerNorm(C_in,eps=1e-5,data_format="channels_first"),
                nn.Conv2d(C_in, C_in, kernel_size=2,stride=2, bias=False)
            ])


        layers.extend([
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in * 2, kernel_size=1, padding=0, bias=False),
            LayerNorm(C_in * 2,eps=1e-5,data_format="channels_first"),
            nn.Conv2d(C_in * 2, C_in, kernel_size=1,padding=0, bias=False),

            # == stacked ==
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in * 2, kernel_size=1, padding=0, bias=False),
            LayerNorm(C_in * 2,eps=1e-5,data_format="channels_first"),
            nn.Conv2d(C_in * 2, C_out, kernel_size=1,padding=0, bias=False)
            ])
        self.op = nn.Sequential(*layers)


    def forward(self, x):
        return self.op(x)



class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1,
                                stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1,
                                stride=2, padding=0, bias=False)
        #self.ln = LayerNorm(C_in,eps=1e-5,data_format="channels_first")
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        if x.shape[2] % 2 == 0:
            out = torch.cat(
                [self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        else:
            x2 = F.pad(x[:, :, 1:, 1:], (0, 1, 0, 1), mode='constant', value=0)
            out = torch.cat([self.conv_1(x), self.conv_2(x2)], dim=1)
        out = self.bn(out)
        return out


class SepConv_new_conv(nn.Module):
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv_new_conv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in*2, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in*2, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in*2, C_in, kernel_size=kernel_size, stride=1,
                      padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)