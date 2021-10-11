# fmt: off
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (
    Distribution, Independent, MixtureSameFamily, MultivariateNormal, Normal
)
from torch.distributions.categorical import Categorical

from deep_traffic_generation.core.abstract import LSR


# fmt:on
class CustomMSF(MixtureSameFamily):
    """MixtureSameFamily with `rsample()` method for reparametrization.

    Args:
        mixture_distribution (Categorical): Manages the probability of
            selecting component. The number of categories must match the
            rightmost batch dimension of the component_distribution.
        component_distribution (Distribution): Define the distribution law
            followed by the components. Right-most batch dimension indexes
            component.
    """

    def rsample(self, sample_shape=torch.Size()):
        """Generates a sample_shape shaped reparameterized sample or
        sample_shape shaped batch of reparameterized samples if the
        distribution parameters are batched.

        Method:

            - Apply `Gumbel Sotmax
              <https://pytorch.org/docs/stable/generated/torch.nn.functional.gumbel_softmax.html>`_
              on component weights to get a one-hot tensor;
            - Sample using rsample() from the component distribution;
            - Use the one-hot tensor to select samples.

        .. note::
            The component distribution of the mixture should implements a
            rsample() method.

        .. warning::
            Further studies should be made on this method. It is highly
            possible that this method is not correct.
        """
        assert (
            self.component_distribution.has_rsample
        ), "component_distribution attribute should implement rsample() method"

        weights = self.mixture_distribution._param
        comp = nn.functional.gumbel_softmax(weights, hard=True).unsqueeze(-1)
        samples = self.component_distribution.rsample(sample_shape)
        return (comp * samples).sum(dim=1)


class NormalLSR(LSR):
    """DEPRECATED: use GaussianMixtureLSR with 1 component."""

    def __init__(self, input_dim: int, out_dim: int, fix_prior: bool = True):
        super().__init__()

        self.z_loc = nn.Linear(input_dim, out_dim)
        self.z_log_var = nn.Linear(input_dim, out_dim)

        self.out_dim = out_dim
        self.dist = Normal

        self.prior_loc = nn.Parameter(
            torch.zeros((1, out_dim)), requires_grad=not fix_prior
        )
        self.prior_log_var = nn.Parameter(
            torch.zeros((1, out_dim)), requires_grad=not fix_prior
        )
        self.register_parameter("prior_loc", self.prior_loc)
        self.register_parameter("prior_log_var", self.prior_log_var)

    def forward(self, hidden) -> Distribution:
        loc = self.z_loc(hidden)
        log_var = self.z_log_var(hidden)
        return self.dist(loc, (log_var / 2).exp())

    def dist_params(self, p: Normal) -> List[torch.Tensor]:
        return [p.loc, p.scale]

    def get_posterior(self, dist_params: List[torch.Tensor]) -> Normal:
        return self.dist(dist_params[0], dist_params[1])

    def get_prior(self, batch_size) -> Normal:
        return self.dist(
            self.prior_loc.expand(batch_size, -1),
            (self.prior_log_var.expand(batch_size, -1) / 2).exp(),
        )


class GaussianMixtureLSR(LSR):
    """Gaussian Mixture Latent Space Regularization.

    .. note::
        It uses a distribution class built on top of MixtureSameFamily to add
        a rsample() method.

        .. autoclass:: deep_traffic_generation.core.lsr.CustomMSF
            :members: rsample

    Args:
        input_dim (int): size of each input sample.
        out_dim (int):size of each output sample.
        n_components (int, optional): Number of components in the Gaussian
            Mixture. Defaults to ``1``.
        fix_prior (bool, optional): Whether to optimize the prior distribution.
            Defaults to ``True``.
    """

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        n_components: int = 1,
        fix_prior: bool = True,
    ):
        super().__init__(input_dim, out_dim, fix_prior)

        self.n_components = n_components

        self.dist = CustomMSF
        self.comp = Normal
        self.mix = Categorical

        # neural networks layers
        self.z_locs = nn.ModuleList(
            [nn.Linear(input_dim, out_dim) for _ in range(n_components)]
        )
        self.z_log_vars = nn.ModuleList(
            [nn.Linear(input_dim, out_dim) for _ in range(n_components)]
        )
        self.z_weights = nn.Linear(input_dim, n_components)

        # prior parameters
        self.prior_weights = nn.Parameter(
            torch.ones((1, n_components)), requires_grad=not fix_prior
        )
        means = torch.linspace(-1, 1, steps=n_components + 2)[1:-1].unsqueeze(
            -1
        )
        self.prior_locs = nn.Parameter(
            means * torch.zeros((1, n_components, out_dim)),
            requires_grad=not fix_prior,
        )
        # fmt: off
        self.prior_log_vars = nn.Parameter(
            torch.zeros((1, n_components, out_dim)),
            requires_grad=not fix_prior
        )
        # fmt: on

        self.register_parameter("prior_weights", self.prior_weights)
        self.register_parameter("prior_locs", self.prior_locs)
        self.register_parameter("prior_log_vars", self.prior_log_vars)

    def forward(self, hidden: torch.Tensor) -> Distribution:
        """[summary]

        Args:
            hidden (torch.Tensor): [description]

        Returns:
            Distribution: [description]
        """
        locs = torch.cat(
            [
                self.z_locs[n](hidden).unsqueeze(1)
                for n in range(self.n_components)
            ],
            dim=1,
        )
        log_vars = torch.cat(
            [
                self.z_log_vars[n](hidden).unsqueeze(1)
                for n in range(self.n_components)
            ],
            dim=1,
        )
        w = self.z_weights(hidden)
        scales = (log_vars / 2).exp()

        return self.dist(
            self.mix(logits=w), Independent(self.comp(locs, scales), 1)
        )

    def dist_params(self, p: MixtureSameFamily) -> Tuple:
        return (
            p.mixture_distribution.logits,
            p.component_distribution.base_dist.loc,
            p.component_distribution.base_dist.scale,
        )

    def get_posterior(self, dist_params: Tuple) -> MixtureSameFamily:
        return self.dist(
            self.mix(logits=dist_params[0]),
            Independent(self.comp(dist_params[1], dist_params[2]), 1),
        )

    def get_prior(self, batch_size: int) -> MixtureSameFamily:
        return self.dist(
            self.mix(logits=self.prior_weights.expand(batch_size, -1)),
            Independent(
                self.comp(
                    self.prior_locs.expand(batch_size, -1, -1),
                    (self.prior_log_vars.expand(batch_size, -1, -1) / 2).exp(),
                ),
                1,
            ),
        )


class MultivariateNormalLSR(LSR):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        fix_prior: bool = True,
    ):
        super().__init__()

        self.out_dim = out_dim
        self.dist = MultivariateNormal

        self.z_loc = nn.Linear(input_dim, out_dim)

        # Sigma = wI + aa' (Cholesky decomposition)
        self.a = nn.Linear(input_dim, out_dim)
        self.w = nn.Linear(input_dim, out_dim)

        self.prior_loc = nn.Parameter(
            torch.zeros((1, out_dim)), requires_grad=not fix_prior
        )
        self.prior_cov = nn.Parameter(
            torch.eye(out_dim), requires_grad=not fix_prior
        )
        self.register_parameter("prior_loc", self.prior_loc)
        self.register_parameter("prior_cov", self.prior_cov)

    def forward(self, hidden: torch.Tensor):
        loc = self.z_loc(hidden)

        Id = (
            torch.eye(self.out_dim, device=hidden.device)
            .unsqueeze(0)
            .repeat(hidden.size(0), 1, 1)
        )
        w = F.relu(self.w(hidden)).unsqueeze(-1) + 0.5
        a = F.tanh(self.a(hidden)).unsqueeze(-1)
        cov = Id * w + torch.matmul(a, torch.transpose(a, dim0=-2, dim1=-1))
        scale = torch.linalg.cholesky(cov)
        return MultivariateNormal(loc, scale_tril=scale)

    def dist_params(self, p: MultivariateNormal) -> List[torch.Tensor]:
        return [p.loc, p.covariance_matrix]

    def get_posterior(
        self, dist_params: List[torch.Tensor]
    ) -> MultivariateNormal:
        return self.dist(dist_params[0], covariance_matrix=dist_params[1])

    def get_prior(self, batch_size: int) -> MultivariateNormal:
        return self.dist(
            self.prior_loc.expand(batch_size, -1),
            self.prior_cov.expand(batch_size, -1, -1),
        )
