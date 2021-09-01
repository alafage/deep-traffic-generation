import torch

from deep_traffic_generation.core.datasets import TrafficDataset


def test_data_shape(trajectory_data):
    dataset = TrafficDataset(
        trajectory_data,
        features=["x", "y", "altitude", "timedelta"],
        shape="linear",
    )
    assert dataset.input_dim == 120
    assert dataset.seq_len == 30
    datapoint, _, _ = dataset[0]
    assert datapoint.size() == torch.Size([120])
