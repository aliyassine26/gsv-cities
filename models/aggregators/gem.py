import torch
import torch.nn.functional as F
import torch.nn as nn


class GeMPool(nn.Module):
    """Implementation of GeM as in https://github.com/filipradenovic/cnnimageretrieval-pytorch
    we add flatten and norm so that we can use it as one aggregation layer.
    """

    def __init__(self, p=3, eps=1e-6):
        """
        Initializes the GeM pooling layer with the given parameters.

        Parameters:
            p (float, optional): The exponent value for the power operation in the GeM pooling. Defaults to 3.
            eps (float, optional): The small constant added to the input tensor to avoid division by zero. Defaults to 1e-6.

        Explanation:
            p is a trainable parameter, for default value of 3, p = torch.tensor([3.0], requires_grad=True)
            eps is a small constant used to avoid division by zero (numerical stability).
        """

        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        """
        Performs the forward pass of the GeM pooling layer.

        Parameters:
            x (torch.Tensor): The input tensor to be processed.

        Returns:
            torch.Tensor: The normalized output tensor after applying the GeM pooling operation.

        Explanation:
            x.clamp(min=self.eps) is used to ensure all elements are at least eps.
            x.clamp(min=self.eps).pow(self.p) is used to apply the power operation to the input tensor.
            F.avg_pool2d(...): Applies 2D average pooling over the input. 
            The size for pooling is set to the height and width of the input feature map 
            (x.size(-2) and x.size(-1)), effectively taking the average over all spatial dimensions.
            .pow(1./self.p): Applies the inverse of the power applied earlier, effectively calculating the p-th root of the average of the elevated pixel values, thus computing the generalized mean.

            x.flatten(1): Flattens the pooled tensor starting from the first dimension, converting it into a 1D vector per example in the batch.
            F.normalize(x, p=2, dim=1): Normalizes these vectors to have unit norm, which is a common practice in retrieval systems to measure similarity using cosine distance.
            - p is the degree of the norm, which is set to 2 for L2 normalization.
            - dim=1 specifies the dimension along which the normalization is applied, which is the batch dimension in this case.
        """
        x = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                         (x.size(-2), x.size(-1))).pow(1./self.p)
        x = x.flatten(1)
        return F.normalize(x, p=2, dim=1)
