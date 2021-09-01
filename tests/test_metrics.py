import torch

from deep_traffic_generation.core.losses import sdtw_loss


def test_npa_loss() -> None:
    torch.manual_seed(42)
    ...


def test_sdtw_loss() -> None:
    torch.manual_seed(42)
    batch_size, len_x, len_y, dims = 8, 34, 30, 4
    x = torch.rand((batch_size, len_x, dims), requires_grad=True)
    y = torch.rand((batch_size, len_y, dims), requires_grad=True)
    loss = sdtw_loss(x, y, reduction="none")
    assert loss.size() == torch.Size([8])
    loss = sdtw_loss(x, y, reduction="mean")
    assert loss.item() == 14.862661361694336
