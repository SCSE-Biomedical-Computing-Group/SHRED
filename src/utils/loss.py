import torch


def reduce(x, reduction="none"):
    if reduction == "none":
        return x
    elif reduction == "mean":
        return x.mean()
    elif reduction == "sum":
        return x.sum()
    else:
        raise TypeError("invalid reduction: {}".format(reduction))


def kl_divergence_loss(
    mu1: torch.Tensor,
    var1: torch.Tensor,
    mu2: torch.Tensor,
    var2: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    kl = 0.5 * (var2.log() - var1.log() + (var1 + (mu1 - mu2) ** 2) / var2 - 1)
    kl = kl.sum(dim=1)
    return reduce(kl, reduction)


def entropy_loss(
    pred_y: torch.Tensor, reduction: str = "mean",
) -> torch.Tensor:
    eps = 1e-12
    pred_y = torch.maximum(pred_y, torch.tensor(eps))
    uni_dist = torch.ones(pred_y.size(0), device=pred_y.device) / pred_y.size(1)
    max_entropy = -uni_dist.log()
    entropy = -pred_y * pred_y.log()
    entropy = entropy.sum(dim=1)
    return reduce(max_entropy - entropy, reduction=reduction)
