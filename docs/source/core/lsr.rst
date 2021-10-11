Latent Space Regularization
===========================

We introduce a new abstract class LSR (Latent Space Regularization) to convert
some hidden vector into a 
`Distribution <https://pytorch.org/docs/stable/distributions.html#distribution>`_
object.

In this project this will be used for Variational Autoencoders. At this time we
only provide a class for implementing a Gaussian Mixture.

.. autoclass:: deep_traffic_generation.core.abstract.LSR
    :members:
    :no-undoc-members:
    :show-inheritance:

.. autoclass:: deep_traffic_generation.core.GaussianMixtureLSR