Installation
============

The deep-traffic-generation library relies on `traffic
<https://traffic-viz.github.io/>`_ and `pytorch <https://pytorch.org/>`_
libraries.

To install the project, follow the instruction below:

.. parsed-literal::

    # create a new python environment for traffic
    conda create -n traffic -c conda-forge python=3.9 traffic
    conda activate traffic

    # clone project
    git clone https://github.com/alafage/deep-traffic-generation

    # install project
    cd deep-traffic-generation
    pip install .