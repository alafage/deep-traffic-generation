# fmt: off
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.optim import Adam
from sklearn.preprocessing import MinMaxScaler
from traffic.core.projection import EuroPP

from deep_traffic_generation.core.builders import (
    CollectionBuilder, IdentifierBuilder, TimestampBuilder
)
from deep_traffic_generation.core.datasets import TrafficDataset
from deep_traffic_generation.core.utils import (
    get_dataloaders, traffic_from_data
)


# fmt: on
# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(z|x)
class Encoder(nn.Module):
    def __init__(self, x_dim, hidden_dim, z_dim):
        super().__init__()
        self.x_dim = x_dim
        # setup the three linear transformations used
        self.fc1 = nn.Linear(x_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the sample x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.reshape(-1, self.x_dim)
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale


# define the PyTorch module that parameterizes the
# observation likelihood p(x|z)
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, x_dim):
        super().__init__()
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, x_dim)
        self.fc22 = nn.Linear(hidden_dim, x_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))
        # return the parameters for the output Normal
        # each is of size batch_size x x_dim
        x_loc = self.fc21(hidden)
        return x_loc


# define a PyTorch module for the VAE
class VAE(nn.Module):
    def __init__(
        self,
        x_dim,
        hidden_dim=400,
        z_dim=20,
        device="cuda" if torch.cuda.is_available() else "cpu",
        sigma=0.005,
    ):
        super().__init__()
        self.encoder = Encoder(x_dim, hidden_dim, z_dim)
        self.decoder = Decoder(z_dim, hidden_dim, x_dim)
        if torch.cuda.is_available():
            self.to(device)
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.sigma = sigma

    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # sample from prior
            # (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            x_loc = self.decoder(z)
            pyro.sample(
                "pred",
                dist.Normal(
                    x_loc, torch.ones_like(x_loc) * self.sigma
                ).to_event(1),
            )
            # score against actual data samples
            pyro.sample(
                "obs",
                dist.Normal(
                    x_loc, torch.ones_like(x_loc) * self.sigma
                ).to_event(1),
                obs=x.reshape(-1, self.x_dim),
            )

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    # define a helper function for reconstructing trajectories
    def reconstruct_traj(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_traj = self.decoder(z)
        return loc_traj


def train(svi, train_loader, device):
    epoch_loss = 0.0
    for x in train_loader:
        x = x[0]
        if torch.cuda.is_available():
            x = x.to(device)
        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step(x)

    # return epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train


def evaluate(svi, test_loader, device):
    test_loss = 0.0
    for x in test_loader:
        x = x[0]
        if torch.cuda.is_available():
            x = x.to(device)
        # compute ELBO estimate and accumulate loss
        test_loss += svi.evaluate_loss(x)
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test


def main():
    # clear param store
    pyro.clear_param_store()

    # ------------
    # args
    # ------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        dest="data_path",
        type=Path,
        default=Path("./data/denoised_v3.pkl").absolute(),
    )
    parser.add_argument(
        "--features",
        dest="features",
        nargs="+",
        default=["latitude", "longitude", "altitude", "timedelta"],
    )
    parser.add_argument("--gpu", dest="gpu", type=str, default="1")
    parser.add_argument(
        "--init_features",
        dest="init_features",
        nargs="+",
        default=[],
    )
    parser.add_argument("--lr", dest="lr", type=float, default=1e-3)
    parser.add_argument(
        "--max_epochs", dest="max_epochs", type=int, default=200
    )
    parser.add_argument(
        "--model_path",
        dest="model_path",
        type=Path,
        default=Path("./models/vae.pth").absolute(),
    )
    parser.add_argument(
        "--vf",
        dest="val_frequency",
        default=5,
        type=int,
        help="how often we evaluate the validation set",
    )
    parser.add_argument(
        "--train_ratio", dest="train_ratio", type=float, default=0.8
    )
    parser.add_argument(
        "--val_ratio", dest="val_ratio", type=float, default=0.2
    )
    parser.add_argument(
        "--batch_size", dest="batch_size", type=int, default=1000
    )
    parser.add_argument(
        "--test_batch_size",
        dest="test_batch_size",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    # ------------
    # data
    # ------------
    dataset = TrafficDataset(
        args.data_path,
        features=args.features,
        init_features=args.init_features,
        scaler=MinMaxScaler(feature_range=(-1, 1)),
        mode="linear",
    )

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset,
        args.train_ratio,
        args.val_ratio,
        args.batch_size,
        args.test_batch_size,
    )

    # ------------
    # model
    # ------------
    vae = VAE(800, device=device)

    # ------------
    # training
    # TODO: should keep best model on validation set
    # ------------
    adam_args = {"lr": args.lr}
    optimizer = Adam(adam_args)

    # setup the inference algorithm
    svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

    train_elbo = []
    val_elbo = []
    # training loop
    for epoch in range(args.max_epochs):
        total_epoch_loss_train = train(svi, train_loader, device)
        train_elbo.append(-total_epoch_loss_train)

        if epoch % args.val_frequency == 0:
            # report test diagnostics
            total_epoch_loss_val = evaluate(svi, val_loader, device)
            val_elbo.append(-total_epoch_loss_val)
            print(
                "[epoch %03d] average test loss: %.4f"
                % (epoch, total_epoch_loss_val)
            )

    # ------------
    # testing
    # ------------
    total_epoch_loss_test = evaluate(svi, test_loader, device)
    print(f"Loss on test set: {total_epoch_loss_test}")

    # ------------
    # saving
    # ------------
    torch.save(vae.state_dict(), args.model_path)

    # ------------
    # plotting
    # ------------
    vae.eval()

    inputs, _, info = next(iter(test_loader))
    outputs = vae.reconstruct_traj(inputs.to(device))

    predictive = Predictive(vae.model, num_samples=1, return_sites=("pred",))
    p_noise = torch.distributions.Normal(torch.zeros(800), torch.ones(800))
    samples = predictive(p_noise.rsample(torch.Size([5])).to(device))
    # print(samples)
    # print(inputs.size())
    print(samples["pred"].size())

    generated = samples["pred"].squeeze(0)[:5].cpu().numpy()

    inputs = inputs[:5].cpu().numpy()
    info = info[:5].cpu().numpy()
    outputs = outputs[:5].detach().cpu().numpy()

    data = np.concatenate((inputs, outputs), axis=0)

    if dataset.scaler is not None:
        data = dataset.scaler.inverse_transform(data)
        generated = dataset.scaler.inverse_transform(generated)

    if len(args.init_features) > 0:
        info = np.repeat(info, data.shape[0], axis=0)
        data = np.concatenate((info, data), axis=1)

    builder = CollectionBuilder(
        [
            IdentifierBuilder(10, 200),
            TimestampBuilder(),
            # LatLonBuilder(build_from="azgs"),
        ]
    )
    builder2 = CollectionBuilder(
        [
            IdentifierBuilder(5, 200),
            TimestampBuilder(),
        ]
    )
    features = ["track" if "track" in f else f for f in args.features]
    # build traffic
    traffic = traffic_from_data(
        data,
        features,
        args.init_features,
        builder=builder,
    )

    traffic_g = traffic_from_data(
        generated, features, args.init_features, builder=builder2
    )

    fig = plt.figure()
    with plt.style.context("traffic"):
        ax = plt.axes(projection=EuroPP())
        traffic_g.plot(ax)
    fig.savefig("img/generated.png")

    fig = plt.figure()
    with plt.style.context("traffic"):
        ax = plt.axes(projection=EuroPP())
        traffic.query("flight_id < '5'").plot(ax)
    fig.savefig("img/originals.png")

    fig = plt.figure()
    with plt.style.context("traffic"):
        ax = plt.axes(projection=EuroPP())
        traffic.query("flight_id >= '5'").plot(ax)
    fig.savefig("img/reconstructed.png")


if __name__ == "__main__":
    # assert pyro.__version__.startswith("1.7.0")
    main()
