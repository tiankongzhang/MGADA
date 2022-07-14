from torch.autograd import Function


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone() * (-ctx.lambd), None


grad_reverse = GradReverse.apply
