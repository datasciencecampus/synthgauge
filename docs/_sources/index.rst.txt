.. SynthGauge documentation master file, created by
   sphinx-quickstart on Tue Jan 18 08:52:26 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: _static/logo.png
   :alt: SynthGauge Logo
   :align: center

Welcome to the SynthGauge Documentation
=======================================

SynthGauge is a Python library to evaluate synthetically generated data.

The library provides a range of metrics and visualisations for assessing and
comparing distributions of features between real and synthetic data.

At its core is the `Evaluator` class, which provides a consistent interface for
applying a range of metrics to the data. By creating several `Evaluator`
instances, you can easily evaluate synthetic data generated from a range of
methods in a consistent and comparable manner.

-------

Use the menu on the left to navigate the documentation. The source code for
SynthGauge is available on the Office for National Statistics Data Science
Campus GitHub `page <https://github.com/datasciencecampus/synthgauge>`_.

.. toctree::
   :hidden:

   Home <self>

.. toctree::
   :maxdepth: 1

   Example notebook <demo>
   autoapi/index


Indices
=======

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
