Installation
============

The deep-traffic-generation library relies on `traffic
<https://traffic-viz.github.io/>`_ and `pytorch <https://pytorch.org/>`_
libraries.

.. plot::
    :include-source: true

    from traffic.data.samples import switzerland
    from traffic.core.projection import EuroPP
    import matplotlib.pyplot as plt

    with plt.style.context("traffic"):
        ax = plt.axes(projection=EuroPP())
        switzerland.plot(ax, alpha=0.1)