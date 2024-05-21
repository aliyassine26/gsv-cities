import torch
import torch.nn.functional as F
import torch.nn as nn


class ConvAP(nn.Module):
    """Implementation of ConvAP as of https://arxiv.org/pdf/2210.10239.pdf

    Args:
        in_channels (int): number of channels in the input of ConvAP
        out_channels (int, optional): number of channels that ConvAP outputs. Defaults to 512.
        s1 (int, optional): spatial height of the adaptive average pooling. Defaults to 2.
        s2 (int, optional): spatial width of the adaptive average pooling. Defaults to 2.
    """

    def __init__(self, in_channels, out_channels=512, s1=2, s2=2):
        """
        Initializes the ConvAP class with the specified parameters.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int, optional): The number of output channels. Defaults to 512.
            s1 (int, optional): The spatial height of the adaptive average pooling. Defaults to 2.
            s2 (int, optional): The spatial width of the adaptive average pooling. Defaults to 2.
        """
        super(ConvAP, self).__init__()
        self.channel_pool = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True)
        self.AAP = nn.AdaptiveAvgPool2d((s1, s2))

    def forward(self, x):
        """
        Performs the forward pass of the ConvAP module.

        Parameters:
            x (torch.Tensor): The input tensor to be processed.

        Returns:
            torch.Tensor: The normalized output tensor after processing through the channel pool and adaptive average pooling.
        """
        x = self.channel_pool(x)
        x = self.AAP(x)
        x = F.normalize(x.flatten(1), p=2, dim=1)
        return x


def print_shape(module, input, output):
    # This function prints the output shape of the layer
    print(f'{module.__class__.__name__} output shape: {output.shape}')


if __name__ == '__main__':
    # x = torch.randn(4, 2048, 10, 10)
    # m = ConvAP(2048, 512)
    # r = m(x)
    # print(r.shape)

    model = ConvAP(2048, 512)

    # Registering hooks on each layer
    for layer in model.children():
        layer.register_forward_hook(print_shape)

    # Create a dummy input tensor
    input_tensor = torch.randn(3, 2048, 32, 32)

    # Perform a forward pass through the model
    output = model(input_tensor)

    print(output.shape)
