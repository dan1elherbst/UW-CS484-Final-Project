from collections import OrderedDict
from typing import Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MonoDepthModel(nn.Module):
    """
    Implementation of the model for self-supervised monocular depth estimation intruduced
    by Clément Godard, Oisin Mac Aodha, Gabriel J. Brostow in

    "Unsupervised Monocular Depth Estimation with Left-Right Consistency" 
    (https://arxiv.org/pdf/1609.03677.pdf).

    This class only implements the non-ResNet variant for the endoder of this model. Some 
    parts of the implementation are inspired by the reference implementation of the 
    original paper, available at https://github.com/mrharicot/monodepth, which is based on
    TensorFlow while this project relies on PyTorch.
    """

    def __init__(self) -> None:
        """
        Initialize the model.
        """
        super().__init__()

        # Encoder

        self.conv_1 = ConvBlock(3, 32, 7)
        self.conv_2 = ConvBlock(32, 64, 5)
        self.conv_3 = ConvBlock(64, 128, 3)
        self.conv_4 = ConvBlock(128, 256, 3)
        self.conv_5 = ConvBlock(256, 512, 3)
        self.conv_6 = ConvBlock(512, 512, 3)
        self.conv_7 = ConvBlock(512, 512, 3)

        # Decoder

        self.upconv_7 = UpConv(512, 512)
        self.iconv_7 = IConv(1024, 512)

        self.upconv_6 = UpConv(512, 512)
        self.iconv_6 = IConv(1024, 512)

        self.upconv_5 = UpConv(512, 256)
        self.iconv_5 = IConv(512, 256)

        self.upconv_4 = UpConv(256, 128)
        self.iconv_4 = IConv(256, 128)
        self.disp_4 = Disp(128)

        self.upconv_3 = UpConv(128, 64)
        self.iconv_3 = IConv(130, 64)
        self.disp_3 = Disp(64)

        self.upconv_2 = UpConv(64, 32)
        self.iconv_2 = IConv(66, 32)
        self.disp_2 = Disp(32)

        self.upconv_1 = UpConv(32, 16)
        self.iconv_1 = IConv(18, 16)
        self.disp_1 = Disp(16)


    def forward(
        self, 
        x: Tensor,
    ) -> Union[Tensor, tuple[Tensor, Tensor, Tensor, Tensor]]:
        """
        Forward pass of the model.

        Args:
            x (tensor [N, C, H, W]):
                Images for which to perform a forward pass of the model. H and W should be
                multiples of 128 for the downscaling to work.
        
        Returns:
            tensor or tuple of tensors:
                Right and left disparity maps for different scales of the iamge that were 
                computed during the forward pass and are used for prediction. Each item of
                the tuple contains a tensor of shape [N, 2, H, W] where the first 
                dimension (apart from batch size) indicates right or left, and H and W are
                down-scaled by a multiplicative factor of 1, 2, 4, and 8. Values of the 
                disparity maps are between 0 and 0.3. If called when the model is in 
                evaluation mode, just the full resolution disparity map is returned.
        """

        # Encoder

        x_conv_1 = self.conv_1(x)
        x_conv_2 = self.conv_2(x_conv_1)
        x_conv_3 = self.conv_3(x_conv_2)
        x_conv_4 = self.conv_4(x_conv_3)
        x_conv_5 = self.conv_5(x_conv_4)
        x_conv_6 = self.conv_6(x_conv_5)
        x_conv_7 = self.conv_7(x_conv_6)

        # Decoder

        x_upconv_7 = self.upconv_7(x_conv_7)
        x_concat_7 = torch.concat((x_upconv_7, x_conv_6), -3)
        x_iconv_7 = self.iconv_7(x_concat_7)

        x_upconv_6 = self.upconv_6(x_iconv_7)
        x_concat_6 = torch.concat((x_upconv_6, x_conv_5), -3)
        x_iconv_6 = self.iconv_6(x_concat_6)

        x_upconv_5 = self.upconv_5(x_iconv_6)
        x_concat_5 = torch.concat((x_upconv_5, x_conv_4), -3)
        x_iconv_5 = self.iconv_5(x_concat_5)

        x_upconv_4 = self.upconv_4(x_iconv_5)
        x_concat_4 = torch.concat((x_upconv_4, x_conv_3), -3)
        x_iconv_4 = self.iconv_4(x_concat_4)
        x_disp_4 = self.disp_4(x_iconv_4)
        x_udisp_4 = F.interpolate(x_disp_4, scale_factor=2)

        x_upconv_3 = self.upconv_3(x_iconv_4)
        x_concat_3 = torch.concat((x_upconv_3, x_conv_2, x_udisp_4), -3)
        x_iconv_3 = self.iconv_3(x_concat_3)
        x_disp_3 = self.disp_3(x_iconv_3)
        x_udisp_3 = F.interpolate(x_disp_3, scale_factor=2)

        x_upconv_2 = self.upconv_2(x_iconv_3)
        x_concat_2 = torch.concat((x_upconv_2, x_conv_1, x_udisp_3), -3)
        x_iconv_2 = self.iconv_2(x_concat_2)
        x_disp_2 = self.disp_2(x_iconv_2)
        x_udisp_2 = F.interpolate(x_disp_2, scale_factor=2)

        x_upconv_1 = self.upconv_1(x_iconv_2)
        x_concat_1 = torch.concat((x_upconv_1, x_udisp_2), 1)
        x_iconv_1 = self.iconv_1(x_concat_1)
        x_disp_1 = self.disp_1(x_iconv_1)

        # Train mode: Return disparity maps on all scales
        if self.training:
            return x_disp_1, x_disp_2, x_disp_3, x_disp_4
        # Eval mode: Return only the full resolution disparity maps
        else:
            return x_disp_1
        
    def blend_disp(self, img_left: Tensor) -> Tensor: 
        """
        Blend disparities for (left) images and a flipped versions of these images during
        test time in order to get accurate disparity estimates in border regions to the 
        left and right of the images. Wraps around the forward pass of the model, i.e.
        two forward passes are carried out if this method is called.

        Args:
            x (tensor [N, C, H, W] or [C, H, W]):
                Images for which to perform a forward pass of the model. H and W should be
                multiples of 128 for the downscaling to work.

        Returns: 
            tensor [N, H, W] or [H, W]:
                Blended (left-aligned) disparity maps.
        """

        expand = True if len(img_left.shape) == 3 else False
        if expand:
            img_left = img_left[None, :, :]

        # Compute disparity maps for image and flipped image
        self.eval()
        disp_left = self(img_left)[:, 1]
        disp_left_flip = torch.flip(self(torch.flip(img_left, (3,)))[:, 1], (2,))

        # Compute weights for the two disparity maps
        H, W = disp_left.shape[1:]
        x = torch.linspace(-10, 10, W).expand(H, -1)
        w1, w2 = torch.sigmoid(x), torch.sigmoid(-x)

        # Return blended disparity maps
        disp = w1[None, :, :] * disp_left + w2[None, :, :] * disp_left_flip
        return torch.squeeze(disp) if expand else disp



class ConvBlock(nn.Module):
    """
    Convolutional block for the encoder part of the model consisting of 2 convolutional
    layer with ELU as an activation function.
    """

    def __init__(
        self,
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        stride_a: int = 2,
        stride_b: int = 1,
    ) -> None:
        super().__init__()
        self.padding = np.floor((kernel_size - 1) / 2).astype(np.int32)
        self.model = nn.Sequential(OrderedDict([
            ('conv_a', nn.Conv2d(in_channels=in_channels, 
                                 out_channels=out_channels, 
                                 kernel_size=kernel_size, 
                                 stride=stride_a,
                                 padding=self.padding)),
            ('activation_a', nn.ELU()),
            ('conv_b', nn.Conv2d(in_channels=out_channels, 
                                 out_channels=out_channels, 
                                 kernel_size=kernel_size, 
                                 stride=stride_b,
                                 padding=self.padding)),
            ('activation_b', nn.ELU())
        ]))

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class UpConv(nn.Module):
    """
    Convolutional block for the decoder part that upsamples the data after it has been
    downsampled in by the encoder. Takes ELU as an activation function. 
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        scale_factor: int = 2,
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(OrderedDict([
            ('upsample', nn.Upsample(scale_factor=scale_factor, mode='nearest')),
            ('conv', nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               padding='same')),
            ('activation', nn.ELU())
        ]))

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class IConv(nn.Module):
    """
    Convolutional block for the decoder part that takes a concatenation of intermediate
    outputs from the encoder and the output of the last layer in the decoder as input
    (similar to skip connections).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               padding='same')),
            ('activation', nn.ELU())
        ]))

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class Disp(nn.Module):
    """
    Convolutional block in the decoder which outputs a (potentially downscaled) disparity
    map.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 2,
        kernel_size: int = 3,
        disp_max_ratio: int = 0.3,
    ) -> None:
        super().__init__()
        self.disp_max_ratio = disp_max_ratio
        self.model = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               padding='same')),
            ('activation', nn.Sigmoid())
        ]))

    def forward(self, x: Tensor) -> Tensor:
        return self.disp_max_ratio * self.model(x)
