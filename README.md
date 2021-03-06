<p align="center">
  <img src="images/logo.png" />
</p>

# SynthGauge

SynthGauge is a Python library providing a framework in which to evaluate
synthetically generated data.

The library provides a range of metrics and visualisations for assessing and
comparing distributions of features between real and synthetic data. At its
core is the `Evaluator` class, which provides a consistent interface for
applying a range of metrics to the data. By creating several `Evaluator`
instances, you can easily evaluate synthetic data generated from a range of
methods in a consistent and comparable manner.

## Privacy vs. Utility
:lock: vs. :bar_chart:

When generating synthetic data, there is generally a trade-off between privacy
(i.e. keeping sensitive information private) and utility (i.e. ensuring the
dataset is still fit for purpose).

The metrics included in SynthGauge fall into these categories and work is
continuing to add more metrics.

## Mission Statement

SynthGauge **is** a toolkit providing metrics and visualisations that aid the
user in the assessment of their synthetic data.

SynthGauge **is not** going to make any decisions on behalf of the user. It
won’t specify if one synthetic dataset is better than another. This decision is
dataset- and purpose-dependent so can vary widely from user to user.

Simply, SynthGauge is a **decision-support tool**, not a decision-making tool.

## Getting Started

### Install

The ``synthgauge`` package is not currently available on PyPI. However it can
be installed directly from GitHub. The package is configured using
`pyproject.toml` and `setup.cfg` files which require newer versions of `pip`,
`setuptools` and `wheel`. Be sure to update these if you havent for a while!

```bash
$ python -m pip install --upgrade pip setuptools wheel
$ python -m pip install git+https://github.com/datasciencecampus/synthgauge
```

Import the package and check the version:

```python
>>> import synthgauge
>>> print(synthgauge.__version__)
1.0.0

```
### Usage
To help users get acquainted example Jupyter Notebooks are included in the
:open_file_folder: `synthgauge/examples` directory. New users are encouraged to
work through `demo_notebook.ipynb`.

The following shows an example workflow for evaluating a single real/synthetic
data pair.

```python
>>> import pandas as pd
>>> import synthgauge as sg

>>> # 1. Load real data into a Pandas DataFrame
>>> real = pd.read_csv('synth_eval/real.csv')
>>>
>>> # 2. Load corresponding synthetic data into Pandas Dataframe
>>> synth = pd.read_csv('synth_eval/synth.csv')
>>>
>>> # 3. Instantiate an Evaluator object
>>> ev = sg.Evaluator(real, synth)
>>>
>>> # 4. Explore the data
>>> ev.describe_categorical()
[...]
>>> ev.plot_histograms(figsize=(12,12));
[...]
>>> # 5. Add metrics to compute
>>> ev.add_metric('wasserstein', 'wass-age', feature='age')
>>>
>>> # 6. Evaluate the metrics
>>> eval_results = ev.evaluate()
>>>
>>> # 7. Review results dictionary
>>> print(eval_results)
{'wass-age': 1.46}

```

## Further Help

The API is desribed in detail in the reference documentation available
[here](https://datasciencecampus.github.io/synthgauge/). Please direct any
questions to [datacampus@ons.gov.uk](mailto:datacampus@ons.gov.uk).

## Contributing

If you encounter any bugs as part of your usage of `synthgauge`, please file an
issue detailing as much information as possible and include a minimal
reproducible example. If you wish to contribute code to the project, please
refer to the [contribution guidelines](CONTRIBUTING.md).
