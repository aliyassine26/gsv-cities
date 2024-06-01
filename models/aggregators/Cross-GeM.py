import torch
import torch.nn as nn
import torch.nn.functional as F
from gem import GeMPool


class CrossGeM(nn.Module):
    """Implementation of CrossGeM pooling layer"""

    def __init__(self, in_channels, patch_size=2, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.gem_pool = GeMPool()
        self.weights = nn.Parameter(torch.ones(patch_size * patch_size))
        self.patch_gem_pools = nn.ModuleList(
            [GeMPool() for _ in range(patch_size * patch_size)])

    def forward(self, x):
        B, C, H, W = x.size()
        assert C % 2 == 0, "The number of channels should be even."

        # Split the tensor into odd and even index channels
        x_odd = x[:, 0::2, :, :]  # Select odd-indexed channels
        x_even = x[:, 1::2, :, :]  # Select even-indexed channels

        # Compute the dot product for corresponding indices for each spatial location
        x_new = x_odd * x_even  # shape (B, C/2, H, W)

        # Now x_new is a 4D tensor of shape (B, C/2, H, W)

        # Compute patches
        S = self.patch_size
        patches = x_new.unfold(2, H // S, H // S).unfold(3, W // S, W // S)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        # shape (B, S*S, C/2, H/S, W/S)
        patches = patches.view(B, S * S, C // 2, H // S, W // S)

        # Apply GeM pooling to each patch
        gem_pooled_patches = []
        for i in range(S * S):
            patch = patches[:, i, :, :, :]
            gem_pooled_patches.append(self.patch_gem_pools[i](patch))
        gem_pooled_patches = torch.stack(
            gem_pooled_patches, dim=1)  # shape (B, S*S, C/2)

        # Apply weights to patches
        weights = F.softmax(self.weights, dim=0)
        weighted_patches = gem_pooled_patches * \
            weights.view(1, -1, 1)  # shape (B, S*S, C/2)
        weighted_patches = weighted_patches.sum(dim=1)  # shape (B, C/2)

        # Apply GeM pooling to the whole feature map
        global_gem = self.gem_pool(x_new)  # shape (B, C/2)

        # Concatenate the weighted patches vector with the global GeM pooled vector
        final_output = torch.cat(
            [weighted_patches, global_gem], dim=1)  # shape (B, C)

        return F.normalize(final_output, p=2, dim=1)


def print_shape(module, input, output):
    # This function prints the output shape of the layer
    print(f'{module.__class__.__name__} output shape: {output.shape}')


if __name__ == '__main__':
    # Example usage
    # Batch of 3, 256 channels, 16x16 spatial dimensions
    # input_tensor = torch.randn(3, 256, 16, 16)

    # model = CrossGeM(in_channels=256, patch_size=2)

    # # Registering hooks on each layer
    # for layer in model.children():
    #     layer.register_forward_hook(print_shape)

    # output = model(input_tensor)
    # print(output.shape)  # Output shape will be (3, 256)

    # Create a dummy input tensor
    x = torch.randn(1, 4, 3, 3)
    # y = torch.randn(1, 4, 3, 3)

    print(x)
    print(x[:, 0::2, 0, 0])
    print(x[:, 1::2, 0, 0])

    x_odd = x[:, 0::2, 0, 0]  # Select odd-indexed channels
    x_even = x[:, 1::2, 0, 0]  # Select even-indexed channels

    # Compute the dot product for corresponding indices for each spatial location
    x_new = x_odd * x_even  # shape (B, C/2, H, W)
    print(x_new)
