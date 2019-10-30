import torch
from torch import nn
from collections import OrderedDict


class Reshape(nn.Module):

    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.reshape(input.size(0), *self.shape)


class AttentionNetwork(nn.Module):

    def __init__(self, conv_params, linear_params, deconv_params):
        super().__init__()
        self.conv_blocks = nn.Sequential(*[
            nn.Sequential(OrderedDict(
                conv=nn.Conv2d(**conv_param),
                norm=nn.BatchNorm2d(conv_param.out_channels),
                actv=nn.ReLU()
            )) for conv_param in conv_params
        ])
        self.linear_blocks = nn.Sequential(*[
            nn.Sequential(OrderedDict(
                linear=nn.Linear(**linear_param),
                unsqueeze=Reshape(linear_param.out_features, 1),
                batch_norm=nn.BatchNorm1d(linear_param.out_features),
                squeeze=Reshape(linear_param.out_features),
                relu=nn.ReLU()
            )) for linear_param in linear_params
        ])
        self.deconv_blocks = nn.Sequential(
            nn.Sequential(*[
                nn.Sequential(OrderedDict(
                    deconv=nn.ConvTranspose2d(**deconv_param),
                    batch_norm=nn.BatchNorm2d(deconv_param.out_channels),
                    relu=nn.ReLU()
                )) for deconv_param in deconv_params[:-1]
            ]),
            nn.Sequential(*[
                nn.Sequential(OrderedDict(
                    deconv=nn.ConvTranspose2d(**deconv_param),
                    batch_norm=nn.BatchNorm2d(deconv_param.out_channels),
                    sigmoid=nn.Sigmoid()
                )) for deconv_param in deconv_params[-1:]
            ])
        )

    def forward(self, input):
        attention = self.conv_blocks(input)
        shape = attention.shape
        attention = attention.reshape(shape[0], -1)
        attention = self.linear_blocks(attention)
        attention = attention.reshape(*shape)
        attention = self.deconv_blocks(attention)
        output = torch.matmul(
            input.reshape(*input.shape[:2], -1).permute(0, 1, 2),
            attention.reshape(*attention.shape[:2], -1).permute(0, 2, 1)
        )
        output = output.reshape(output.size(0), -1)
        return output, attention
