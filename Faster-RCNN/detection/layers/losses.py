import torch
import torch.nn.functional as F


def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from PyTorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()


def sigmoid_focal_loss(inputs, targets, alpha=-1, gamma=2, reduction="mean"):
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def softmax_focal_loss(inputs, targets, gamma=2, reduction='mean'):
    n, num_classes = inputs.shape
    p = F.softmax(inputs, dim=1)  # (N, C)
    targets = F.one_hot(targets, num_classes)  # (N, C)
    probs = (p * targets).sum(dim=1)  # (N, )

    batch_loss = -(1 - probs) ** gamma * torch.log(probs)
    if reduction == 'mean':
        loss = batch_loss.mean()
    elif reduction == 'sum':
        loss = batch_loss.sum()
    elif reduction == 'none':
        loss = batch_loss
    else:
        raise ValueError
    return loss


def l2_loss(inputs, targets, reduction='mean'):
    loss = F.mse_loss(inputs.sigmoid(), targets.to(inputs.dtype).expand_as(inputs), reduction=reduction)
    return loss
