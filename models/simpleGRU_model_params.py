from collections import OrderedDict

import torch.nn as nn

from models.simple_GRU import SimpleGRU


# build model
def get_encoder_params_GRU(input_channel_count, input_shape):
    print(f'[EncoderParams] channel_count:{input_channel_count} Shape:{input_shape}')
    msg = ('Input gets downsampled by a factor of 5,3 and 2 sequentially and subsequently ' 'it gets upsampled. ')
    assert input_shape[0] % 30 == 0, msg
    assert input_shape[1] % 30 == 0, msg
    x, y = input_shape
    encoder_params = [[
        OrderedDict({
            'conv1_leaky_1': [input_channel_count, 8, 7, 5, 1]
        }),
        OrderedDict({
            'conv2_leaky_1': [64, 192, 5, 3, 1]
        }),
        OrderedDict({
            'conv3_leaky_1': [192, 192, 3, 2, 1]
        }),
    ], [
        SimpleGRU(
            input_channel=8,
            num_filter=64,
            h_w=(x // 5, y // 5),
            zoneout=0.0,
            L=13,
            i2h_kernel=(3, 3),
            i2h_stride=(1, 1),
            i2h_pad=(1, 1),
            h2h_kernel=(5, 5),
            h2h_dilate=(1, 1),
            act_type=nn.LeakyReLU(negative_slope=0.2, inplace=True)),
        SimpleGRU(
            input_channel=192,
            num_filter=192,
            h_w=(x // 15, y // 15),
            zoneout=0.0,
            L=13,
            i2h_kernel=(3, 3),
            i2h_stride=(1, 1),
            i2h_pad=(1, 1),
            h2h_kernel=(5, 5),
            h2h_dilate=(1, 1),
            act_type=nn.LeakyReLU(negative_slope=0.2, inplace=True)),
        SimpleGRU(
            input_channel=192,
            num_filter=192,
            h_w=(x // 30, y // 30),
            zoneout=0.0,
            L=9,
            i2h_kernel=(3, 3),
            i2h_stride=(1, 1),
            i2h_pad=(1, 1),
            h2h_kernel=(3, 3),
            h2h_dilate=(1, 1),
            act_type=nn.LeakyReLU(negative_slope=0.2, inplace=True))
    ]]
    return encoder_params


forecaster_params_GRU = [
    [
        OrderedDict({
            'deconv1_leaky_1': [192, 192, 4, 2, 1]
        }),
        OrderedDict({
            'deconv2_leaky_1': [192, 64, 5, 3, 1]
        }),
        OrderedDict({
            'deconv3_leaky_1': [64, 8, 7, 5, 1],
            'conv3_leaky_2': [8, 8, 3, 1, 1],
            'conv3_3': [8, 1, 1, 1, 0]
        }),
    ],
    [
        SimpleGRU(
            input_channel=192,
            num_filter=192,
            h_w=(16, 16),  # For forecaster, this variable is of no use.
            zoneout=0.0,
            L=13,
            i2h_kernel=(3, 3),
            i2h_stride=(1, 1),
            i2h_pad=(1, 1),
            h2h_kernel=(3, 3),
            h2h_dilate=(1, 1),
            act_type=nn.LeakyReLU(negative_slope=0.2, inplace=True)),
        SimpleGRU(
            input_channel=192,
            num_filter=192,
            h_w=(32, 32),  # For forecaster, this variable is of no use.
            zoneout=0.0,
            L=13,
            i2h_kernel=(3, 3),
            i2h_stride=(1, 1),
            i2h_pad=(1, 1),
            h2h_kernel=(5, 5),
            h2h_dilate=(1, 1),
            act_type=nn.LeakyReLU(negative_slope=0.2, inplace=True)),
        SimpleGRU(
            input_channel=64,
            num_filter=64,
            h_w=(96, 96),  # For forecaster, this variable is of no use.
            zoneout=0.0,
            L=9,
            i2h_kernel=(3, 3),
            i2h_stride=(1, 1),
            i2h_pad=(1, 1),
            h2h_kernel=(5, 5),
            h2h_dilate=(1, 1),
            act_type=nn.LeakyReLU(negative_slope=0.2, inplace=True))
    ]
]
