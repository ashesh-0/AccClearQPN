import torch
import torch.nn.functional as F
from torch import nn

# from nowcasting.config import cfg
from core.constants import INPUT_LEN
from models.trajGRU import BaseConvRNN

# self._prefix = prefix
# self._num_filter = num_filter
# self._h2h_kernel = h2h_kernel
# assert (self._h2h_kernel[0] % 2 == 1) and (self._h2h_kernel[1] % 2 == 1), \
#     "Only support odd number, get h2h_kernel= %s" % str(h2h_kernel)
# self._h2h_pad = (h2h_dilate[0] * (h2h_kernel[0] - 1) // 2, h2h_dilate[1] * (h2h_kernel[1] - 1) // 2)
# self._h2h_dilate = h2h_dilate
# self._i2h_kernel = i2h_kernel
# self._i2h_stride = i2h_stride
# self._i2h_pad = i2h_pad
# self._i2h_dilate = i2h_dilate
# self._act_type = act_type


class SimpleGRU(BaseConvRNN):
    # b_h_w: input feature map size
    def __init__(self,
                 input_channel,
                 num_filter,
                 h_w,
                 zoneout=0.0,
                 L=5,
                 i2h_kernel=(3, 3),
                 i2h_stride=(1, 1),
                 i2h_pad=(1, 1),
                 h2h_kernel=(5, 5),
                 h2h_dilate=(1, 1),
                 act_type=nn.LeakyReLU(negative_slope=0.2, inplace=True)):
        super(SimpleGRU, self).__init__(
            num_filter=num_filter,
            h_w=h_w,
            h2h_kernel=h2h_kernel,
            h2h_dilate=h2h_dilate,
            i2h_kernel=i2h_kernel,
            i2h_pad=i2h_pad,
            i2h_stride=i2h_stride,
            act_type=act_type,
            prefix='SimpleGRU')
        self._L = L
        self._zoneout = zoneout

        # reset_gate, update_gate, new_mem
        self.i2h = nn.Conv2d(
            in_channels=input_channel,
            out_channels=self._num_filter * 3,
            kernel_size=self._i2h_kernel,
            stride=self._i2h_stride,
            padding=self._i2h_pad,
            dilation=self._i2h_dilate)

        self.h2h = nn.Conv2d(
            in_channels=self._num_filter,
            out_channels=3 * self._num_filter,
            kernel_size=(5, 5),
            stride=1,
            padding=(2, 2),
            dilation=(1, 1),
        )

    # inputs 和 states 不同时为空
    # inputs: S*B*C*H*W
    def forward(self, inputs=None, states=None, seq_len=None):
        assert seq_len is not None
        if states is None:
            # NOTE: one could also just use shape of inputs to replace use of self._state_height, self._state_width
            states = torch.zeros(
                (inputs.size(1), self._num_filter, self._state_height, self._state_width), dtype=torch.float)
            states = states.type_as(inputs)

        if inputs is not None:
            S, B, C, H, W = inputs.size()
            i2h = self.i2h(torch.reshape(inputs, (-1, C, H, W)))
            i2h = torch.reshape(i2h, (S, B, i2h.size(1), i2h.size(2), i2h.size(3)))
            i2h_slice = torch.split(i2h, self._num_filter, dim=2)

        prev_h = states
        outputs = []
        for i in range(seq_len):

            h2h = self.h2h(prev_h)
            h2h_slice = torch.split(h2h, self._num_filter, dim=1)
            if inputs is not None:
                reset_gate = torch.sigmoid(i2h_slice[0][i, ...] + h2h_slice[0])
                update_gate = torch.sigmoid(i2h_slice[1][i, ...] + h2h_slice[1])
                new_mem = self._act_type(i2h_slice[2][i, ...] + reset_gate * h2h_slice[2])
            else:
                reset_gate = torch.sigmoid(h2h_slice[0])
                update_gate = torch.sigmoid(h2h_slice[1])
                new_mem = self._act_type(reset_gate * h2h_slice[2])

            next_h = update_gate * prev_h + (1 - update_gate) * new_mem
            outputs.append(next_h)
            prev_h = next_h

        return torch.stack(outputs), next_h
