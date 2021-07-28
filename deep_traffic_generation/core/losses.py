import math

import torch


def npa_loss(
    input: torch.Tensor, navpoints: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """The Navigational Point Alignment loss.

    Parameters:
    input: torch.Tensor
        Input tensor should be a 3D Tensor (N, H, L) with 3 features:
        (`x`, `y`, `track`). Note: `track` should be in radians.
    navpoints: torch.Tensor
        Navpoints coordinates, a 2D Tensor (*, 2).
    reduction: string, optional
        Specifies the reduction to apply to the output: `'none'` | `'mean'` |
        `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of
        the output will be divided by the number of the elements in the
        output, `'sum'`: the output will be summed. Default: `'mean'`
    """
    # Compute bearing for each navpoints at each timestamp
    x1 = input[:, 0, :].unsqueeze(2)
    y1 = input[:, 1, :].unsqueeze(2)
    tracks = input[:, 2, :].unsqueeze(2)
    x2 = navpoints[:, 0]
    y2 = navpoints[:, 1]
    # bearings: (N, L, len(navpoints))
    bearings = torch.atan2(x2 - x1, y2 - y1) - torch.Tensor([math.pi / 2]).to(
        input.device
    )
    # mins: (N, L)
    mins, _ = torch.min((tracks - bearings).abs(), dim=2)
    # loss: (N)
    loss = torch.sum(mins, dim=1)

    if reduction == "none":
        return loss
    elif reduction == "mean":
        return torch.mean(loss)
    elif reduction == "sum":
        return torch.sum(loss)
    else:
        raise ValueError(f"Invalid value {reduction} for reduction attribute.")


def custom_loss():
    ...


if "__main__" == __name__:
    input = torch.zeros((10, 3, 20)).requires_grad_(True)
    navpoints = torch.ones((5, 2))
    loss = npa_loss(input, navpoints)
    print(loss.grad_fn)
    print(loss.item())
    loss.backward()
    print(input.grad.size())
