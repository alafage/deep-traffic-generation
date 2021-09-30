Neural Networks
===============

The `deep_traffic_generation` library provides the following deep models:

* Autoencoders:

  - `FCAE <fcae.html>`_ (Fully-Connected Autoencoder);
  - `RAE <rae.html>`_ (Recurrent Autoencoder);
  - `TCAE <tcae.html>`_ (Temporal Convolutional Autoencoder).

* Variational Autoencoders:

  - `FCVAE <fcvae.html>`_ (Fully-Connected Variational Autoencoder);
  - `RVAE <rvae.html>`_ (Recurrent Variational Autoencoder);
  - `TCVAE <tcvae.html>`_ (Temporal Convolutional Variational Autoencoder).


You can train those models from the terminal. You can set up any parameters
of the pytorch-lightning `Trainer 
<https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-in-python-scripts>`_
object using options.

`deep_traffic_generation` adds severals other parameters to set up the
training. We advise you to look for the available parameters of the dataset
(ex: `TrafficDataset arguments <core/traffic_dataset.html#deep_traffic_generation.core.TrafficDataset.add_argparser_args>`_)
or neural networks (ex: `TCVAE arguments <tcvae.html>`_) you use.

Here the list of those options:

* ``--train_ratio``: Represents the proportion of the dataset to use to train
  the model. The other proportion will be used as test set. Default to
  :math:`0.8`.

  Usage:

  .. code-block:: console

    python main.py --train_ratio 0.7

* ``--val_ratio``: Represents the proportion of the train set to use as
  validation set. The other proportion will be the actual train set. Default
  to :math:`0.2`.

  Usage:

  .. code-block:: console

    python main.py --val_ratio 0.7

* ``--batch_size``: Size of batch in train set. Default to :math:`1000`.

  Usage:

  .. code-block:: console

    python main.py --batch_size 1500

* ``--test_batch_size``: Size of batch in both validation and test sets. Default
  to None. In the case it is None, the batch size is the size of the validation
  set.

  Usage:

  .. code-block:: console

    python main.py --test_batch_size 1000

* ``--early_stop``: Define the ``patience`` of the `EarlyStopping Callback 
  <https://pytorch-lightning.readthedocs.io/en/latest/common/early_stopping.html>`_
  monitoring the validation loss.

  Usage:

  .. code-block:: console

    python main.py --early_stop 10

.. toctree::
    :hidden:

    fcae
    rae
    tcae
    fcvae
    rvae
    tcvae