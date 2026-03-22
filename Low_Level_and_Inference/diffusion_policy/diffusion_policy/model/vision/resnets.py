from torchvision import models as vision_models
from robomimic.models.base_nets import ConvBase, CoordConv2d
import torch
import torch.nn as nn
import math
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from robomimic.models.base_nets import ResNet18Conv
class ResNet34Conv(ConvBase):
    """
    A ResNet34 block that can be used to process input images.
    """
    def __init__(
        self,
        input_channel=3,
        pretrained=False,
        input_coord_conv=False,
    ):
        """
        Args:
            input_channel (int): number of input channels for input images to the network.
                If not equal to 3, modifies first conv layer in ResNet to handle the number
                of input channels.
            pretrained (bool): if True, load pretrained weights for all ResNet layers.
            input_coord_conv (bool): if True, use a coordinate convolution for the first layer
                (a convolution where input channels are modified to encode spatial pixel location)
        """
        super(ResNet34Conv, self).__init__()
        net = vision_models.resnet34(pretrained=pretrained)

        if input_coord_conv:
            net.conv1 = CoordConv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif input_channel != 3:
            net.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # cut the last fc layer
        self._input_coord_conv = input_coord_conv
        self._input_channel = input_channel
        self.nets = torch.nn.Sequential(*(list(net.children())[:-2]))

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)
        out_h = int(math.ceil(input_shape[1] / 32.))
        out_w = int(math.ceil(input_shape[2] / 32.))
        return [512, out_h, out_w]

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        return header + '(input_channel={}, input_coord_conv={})'.format(self._input_channel, self._input_coord_conv)

class ResNet50Conv(ConvBase):
    """
    A ResNet34 block that can be used to process input images.
    """
    def __init__(
        self,
        input_channel=3,
        pretrained=False,
        input_coord_conv=False,
    ):
        """
        Args:
            input_channel (int): number of input channels for input images to the network.
                If not equal to 3, modifies first conv layer in ResNet to handle the number
                of input channels.
            pretrained (bool): if True, load pretrained weights for all ResNet layers.
            input_coord_conv (bool): if True, use a coordinate convolution for the first layer
                (a convolution where input channels are modified to encode spatial pixel location)
        """
        super(ResNet50Conv, self).__init__()
        net = vision_models.resnet50(pretrained=pretrained)

        if input_coord_conv:
            net.conv1 = CoordConv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif input_channel != 3:
            net.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # cut the last fc layer
        self._input_coord_conv = input_coord_conv
        self._input_channel = input_channel
        self.nets = torch.nn.Sequential(*(list(net.children())[:-2]))

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)
        out_h = int(math.ceil(input_shape[1] / 32.))
        out_w = int(math.ceil(input_shape[2] / 32.))
        return [2048, out_h, out_w]

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        return header + '(input_channel={}, input_coord_conv={})'.format(self._input_channel, self._input_coord_conv)


def enlarge_resnets(obs_encoder, encoder_backbone):
    resnet_class = ResNet34Conv if encoder_backbone == 'resnet34' else ResNet50Conv
    replace_submodules(
        root_module=obs_encoder,
        predicate=lambda x: isinstance(x, ResNet18Conv),
        func=lambda x: resnet_class(
            input_channel=x._input_channel,
            pretrained=False,
            input_coord_conv=x._input_coord_conv
        ),
        check_bn=False
    )


if __name__ == "__main__":
    from robomimic.models.base_nets import ResNet18Conv
    # simple test
    net18 = ResNet18Conv(input_channel=3, pretrained=False, input_coord_conv=True)
    net34 = ResNet34Conv(input_channel=3, pretrained=False, input_coord_conv=True)
    net50 = ResNet50Conv(input_channel=3, pretrained=False, input_coord_conv=True)
    in_shape = [3, 240, 240]

    x = torch.randn(32, *in_shape)
    out18 = net18(x)
    out34 = net34(x)
    out50 = net50(x)
    print("ResNet18Conv output shape:", out18.shape, " expected: ", [32] + net18.output_shape(in_shape))
    print("ResNet34Conv output shape:", out34.shape, " expected: ", [32] + net34.output_shape(in_shape))
    print("ResNet50Conv output shape:", out50.shape, " expected: ", [32] + net50.output_shape(in_shape))