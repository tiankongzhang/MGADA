import torch
import torch.nn.functional as F


def style_pool2d(input, kernel_size, stride=None, eps=1e-12):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    elif isinstance(stride, int):
        stride = (stride, stride)
    n, c, h, w = input.shape
    new_h, new_w = int((h - kernel_size[0]) / stride[0] + 1), int((w - kernel_size[1]) / stride[1] + 1)

    windows = F.unfold(input, kernel_size=kernel_size, stride=stride)
    windows = windows.view(n, c, kernel_size[0], kernel_size[1], -1)
    var, mean = torch.var_mean(windows, dim=(2, 3), unbiased=False)  # (n, c, new_h * new_w)
    std = torch.sqrt(var + eps)

    windows = torch.cat((mean, std), dim=1)
    windows = windows.view(n, c * 2, new_h, new_w)  # (n, c * 2, new_h, new_w)
    return windows


class StylePool2d(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, eps=1e-12):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = eps

    def forward(self, input):
        return style_pool2d(input, self.kernel_size, self.stride, self.eps)
