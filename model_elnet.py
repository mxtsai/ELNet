### Implementation of ELNet by Maxwell Tsai, Oct 2020
### GitHub Link: https://github.com/mxtsai/ELNet
### Paper Link: https://arxiv.org/abs/2005.02706

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import os
from torch.nn.init import kaiming_uniform_, kaiming_normal_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_deterministic(seed):

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weight_init(m, seed=2, init_type='uniform'):

    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):

        if init_type == 'normal':
            kaiming_normal_(m.weight)
        else:
            kaiming_uniform_(m.weight, a=math.sqrt(5))
    else:
        raise TypeError("cannnot initialize such weights")


def get_norm_layer(channels, norm_type='layer'):
    if norm_type == 'instance':
        layer = nn.GroupNorm(channels, channels)
    elif norm_type == 'batch':
        layer = nn.BatchNorm2d(channels)
    else:
        layer = nn.GroupNorm(1, channels)  # layer norm by default

    return layer


def conv_block(channels, kernel_size, dilation=1, repeats=2, normalization='layer', seed=2, init_type='uniform'):
    """
    :param channels: the input channel amount (same for output)
    :param kernel_size: 2D convolution kernel
    :param dilation: the dialation for the kernels of a conv block
    :param padding: amount of padding
    :param repeats: amount of repeats before added with identity
    :param normalization: the type of multi-slice normalization used
    :param seed: which seed of initial weights to use
    :param init_type: which type of Kaiming Init to use
    :return: nn.Sequential(for the given block)
    """

    conv_list = nn.ModuleList([])

    for i in range(repeats):
        conv2d = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                           dilation=dilation, stride=1, bias=False,
                           padding=(kernel_size + ((dilation - 1) * (kernel_size - 1))) // 2)
        weight_init(conv2d, seed=seed, init_type=init_type)
        conv_list.append(conv2d)

        #   Instance Normalization and Layer Normalization are just variations of Group Normalization
        #   https://pytorch.org/docs/stable/nn.html#groupnorm

        conv_list.append(get_norm_layer(channels, normalization))
        conv_list.append(nn.ReLU())

    return nn.Sequential(*conv_list)


def get_antialiasing_filter(kernel_size):
    """Get an integer specifying the 2D kernel size >>> returns a (1 x 1 x kernel_size x kernel_size)"""

    kernel_dict = {
        1: [[[[1.]]]],

        2: [[[[0.2500, 0.2500],
              [0.2500, 0.2500]]]],

        3: [[[[0.0625, 0.1250, 0.0625],
              [0.1250, 0.2500, 0.1250],
              [0.0625, 0.1250, 0.0625]]]],

        4: [[[[0.0156, 0.0469, 0.0469, 0.0156],
              [0.0469, 0.1406, 0.1406, 0.0469],
              [0.0469, 0.1406, 0.1406, 0.0469],
              [0.0156, 0.0469, 0.0469, 0.0156]]]],

        5: [[[[0.0039, 0.0156, 0.0234, 0.0156, 0.0039],
              [0.0156, 0.0625, 0.0938, 0.0625, 0.0156],
              [0.0234, 0.0938, 0.1406, 0.0938, 0.0234],
              [0.0156, 0.0625, 0.0938, 0.0625, 0.0156],
              [0.0039, 0.0156, 0.0234, 0.0156, 0.0039]]]],

        6: [[[[0.0010, 0.0049, 0.0098, 0.0098, 0.0049, 0.0010],
              [0.0049, 0.0244, 0.0488, 0.0488, 0.0244, 0.0049],
              [0.0098, 0.0488, 0.0977, 0.0977, 0.0488, 0.0098],
              [0.0098, 0.0488, 0.0977, 0.0977, 0.0488, 0.0098],
              [0.0049, 0.0244, 0.0488, 0.0488, 0.0244, 0.0049],
              [0.0010, 0.0049, 0.0098, 0.0098, 0.0049, 0.0010]]]],

        7: [[[[0.0002, 0.0015, 0.0037, 0.0049, 0.0037, 0.0015, 0.0002],
              [0.0015, 0.0088, 0.0220, 0.0293, 0.0220, 0.0088, 0.0015],
              [0.0037, 0.0220, 0.0549, 0.0732, 0.0549, 0.0220, 0.0037],
              [0.0049, 0.0293, 0.0732, 0.0977, 0.0732, 0.0293, 0.0049],
              [0.0037, 0.0220, 0.0549, 0.0732, 0.0549, 0.0220, 0.0037],
              [0.0015, 0.0088, 0.0220, 0.0293, 0.0220, 0.0088, 0.0015],
              [0.0002, 0.0015, 0.0037, 0.0049, 0.0037, 0.0015, 0.0002]]]]

    }

    if kernel_size in kernel_dict:
        return torch.Tensor(kernel_dict[kernel_size])
    else:
        raise ValueError('Unrecognized kernel size')


class BlurPool(nn.Module):

    def __init__(self, channels, stride, filter_size=5):
        super(BlurPool, self).__init__()
        self.channels = channels  # same input and output channels
        self.filter_size = filter_size
        self.stride = stride
        '''Kernel is a 1x5x5 kernel'''

        # repeat tensor from (1 x 1 x fs x fs) >>> (channels x 1 x fs x fs)
        self.kernel = nn.Parameter(get_antialiasing_filter(filter_size).repeat(self.channels, 1, 1, 1),
                                   requires_grad=False)

    def forward(self, x):
        """
        x is a tensor of dimension (batch, in_channels, height, width)
        - assume same input and output channels, and groups = 1
        - CURRENTLY DON'T SUPPORT PADDING
        """

        y = F.conv2d(input=x, weight=self.kernel, stride=self.stride, groups=self.channels)
        return y

    def to(self, dtype):
        self.kernel = self.kernel.to(dtype)
        return self



class ELNet(nn.Module):

    def __init__(self, **kwargs):

        super(ELNet, self).__init__()

        self.K = kwargs.get('K', 4)  # default K for ELNet
        self.norm_type = kwargs.get('norm_type', 'instance')  # default multi-slice normalization
        self.aa_filter = kwargs.get('aa_filter_size', 5)  # default aa-filter configuration
        self.weight_init_type = kwargs.get('weight_init_type', 'normal')  # type of weight initialization
        self.seed = kwargs.get('seed', 2)  # default seed for initialization
        self.num_classes = kwargs.get('num_classes', 2)  # number of classes for ELNet
	self.feature_dropout = kwargs.get('dropout', 0.0)

        make_deterministic(self.seed)

        if isinstance(self.aa_filter, int):
            aa_filter_size = [self.aa_filter] * 5

        self.channel_config = [4 * self.K, 8 * self.K, 16 * self.K, 16 * self.K, 16 * self.K]

        self.conv_1 = nn.Conv2d(1, self.channel_config[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.downsample_1 = BlurPool(channels=self.channel_config[0], stride=2, filter_size=aa_filter_size[0])
        self.norm_1 = get_norm_layer(self.channel_config[0], norm_type=self.norm_type)

        self.conv_2 = conv_block(self.channel_config[0], kernel_size=5, repeats=2, normalization=self.norm_type)
        self.conv_2_to_3 = nn.Conv2d(self.channel_config[0], self.channel_config[1], kernel_size=5, stride=1, padding=2,
                                     bias=False)
        self.downsample_2 = BlurPool(channels=self.channel_config[1], stride=2, filter_size=aa_filter_size[1])

        self.conv_3 = conv_block(self.channel_config[1], kernel_size=3, repeats=2, normalization=self.norm_type)
        self.conv_3_to_4 = nn.Conv2d(self.channel_config[1], self.channel_config[2], kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.downsample_3 = BlurPool(channels=self.channel_config[2], stride=2, filter_size=aa_filter_size[2])

        self.conv_4 = conv_block(self.channel_config[2], kernel_size=3, repeats=1, normalization=self.norm_type)
        self.conv_4_to_5 = nn.Conv2d(self.channel_config[2], self.channel_config[3], kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.downsample_4 = BlurPool(channels=self.channel_config[3], stride=2, filter_size=aa_filter_size[3])

        self.conv_5 = conv_block(self.channel_config[3], kernel_size=3, repeats=1, normalization=self.norm_type)
        self.conv_5_to_6 = nn.Conv2d(self.channel_config[3], self.channel_config[4], kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.downsample_5 = BlurPool(channels=self.channel_config[4], stride=2, filter_size=aa_filter_size[4])

        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.feature_dp = nn.Dropout(p=self.feature_dropout)

        self.fc = nn.Linear(self.channel_config[4], self.num_classes)

        weight_init(self.conv_1, self.seed, self.weight_init_type)
        weight_init(self.conv_2_to_3, self.seed, self.weight_init_type)
        weight_init(self.conv_3_to_4, self.seed, self.weight_init_type)
        weight_init(self.conv_4_to_5, self.seed, self.weight_init_type)
        weight_init(self.conv_5_to_6, self.seed, self.weight_init_type)
        weight_init(self.fc, self.seed, self.weight_init_type)

    def feature_extraction(self, x):
        x = x.permute(1, 0, 2, 3)

        if self.training:

            x = self.downsample_1(F.relu(self.norm_1(self.conv_1(x))))

            x = x + self.conv_2(x)  # skip connection (survival rate 1 for first skip)
            x = self.downsample_2(F.relu(self.conv_2_to_3(x)))

	    x = x + self.conv_3(x)
            x = self.downsample_3(F.relu(self.conv_3_to_4(x)))

            x = x + self.conv_4(x)
            x = self.downsample_4(F.relu(self.conv_4_to_5(x)))

            x = x + self.conv_5(x)
            x = self.downsample_5(F.relu(self.conv_5_to_6(x)))

        else:  # evaluation mode

            x = self.downsample_1(F.relu(self.norm_1(self.conv_1(x)), inplace=True))

            x += self.conv_2(x)  # in-place skip connection (not suitable for training pass)
            x = self.downsample_2(F.relu(self.conv_2_to_3(x), inplace=True))

            x += self.conv_3(x)
            x = self.downsample_3(F.relu(self.conv_3_to_4(x), inplace=True))

            x += self.conv_4(x)  # skip connection
            x = self.downsample_4(F.relu(self.conv_4_to_5(x), inplace=True))

            x += self.conv_5(x)  # skip connection
            x = self.downsample_5(F.relu(self.conv_5_to_6(x), inplace=True))

        feats = nn.AdaptiveMaxPool2d(1)(x)  # get [sx16Kx1x1]
        feats = self.feature_dp(feats)  # performs feature-wise dropout

        return feats

    def forward(self, x):
        feats = self.feature_extraction(x)  # get [sx16Kx1x1]
        feats = feats.squeeze(3)
        feats = feats.permute(2, 1, 0)  # [1x16Kxs]

        # classifier
        feats = self.max_pool(feats).squeeze(2)  # [1x16K]
        scores = self.fc(feats)
        return scores


