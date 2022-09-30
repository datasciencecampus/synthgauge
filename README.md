<p align="center">
  <img src="images/logo.png" />
</p>

# SynthGauge

SynthGauge is a Python library providing a framework in which to evaluate
synthetically generated data.

The library provides a range of metrics and visualisations for assessing and
comparing distributions of features between real and synthetic data. At its
core is the `Evaluator` class, which provides a consistent interface for
assessing two sets of data. By creating several `Evaluator` instances, you can
easily evaluate synthetic data generated from a range of methods in a
consistent and comparable manner.

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

The `synthgauge` package is available on PyPI and can be installed via
`pip` in the standard way:

```bash
$ python -m pip install synthgauge
```

If you'd rather install the package from source, you can do so by first cloning
this repository from GitHub. The `synthgauge` package is configured using
`setup.cfg`, which requires newer versions of `pip`, `setuptools` and `wheel`.
Be sure to update these if you haven't for a while.

```bash
$ cd /path/to/synthgauge
$ python -m pip install --upgrade pip setuptools wheel
$ python -m pip install .
```

Now you're ready to start using the package!

### Usage

To help users get acquainted with the package, an example Jupyter Notebook is
included in the :open_file_folder: `examples` directory. This notebook is
also available in the [package documentation](https://datasciencecampus.github.io/synthgauge/demo.html).

The following shows an example workflow for evaluating a single real-synthetic
dataset pair.

```python
>>> import synthgauge as sg
>>>
>>> # 1. Create or read in some data as a `pandas.DataFrame`
>>> real = sg.datasets.make_blood_types_df(noise=0, nan_prop=0, seed=0)
>>> synth = sg.datasets.make_blood_types_df(noise=1, nan_prop=0, seed=0)
>>>
>>> # 2. Instantiate an Evaluator object
>>> ev = sg.Evaluator(real, synth)
>>>
>>> # 3. Explore the data
>>> ev.describe_numeric()
               count     mean        std    min    25%    50%    75%    max
age_real      1000.0   41.745   7.073472   22.0   37.0   41.0   48.0   62.0
age_synth     1000.0   41.536   9.195829   18.0   35.0   41.0   48.0   68.0
height_real   1000.0  174.976   7.771346  153.0  169.0  176.0  181.0  194.0
height_synth  1000.0  175.266   9.633070  147.0  168.0  176.0  182.0  205.0
weight_real   1000.0   80.014   9.455115   56.0   74.0   80.0   86.0  114.0
weight_synth  1000.0   80.117  11.113452   50.0   72.0   80.0   88.0  118.0
>>> ev.describe_categorical()
                  count unique most_frequent freq
blood_type_real    1000      4             O  384
blood_type_synth   1000      4             A  535
eye_colour_real    1000      3         Brown  577
eye_colour_synth   1000      3         Brown  664
hair_colour_real   1000      4         Brown  435
hair_colour_synth  1000      4         Brown  480
>>> ev.plot_histograms(figsize=(12,12))
<Figure size 1200x1200 with 6 Axes>
>>>
>>> # 4. Add metrics to compute
>>> ev.add_metric('wasserstein', 'wass-age', feature='age')
>>>
>>> # 5. Evaluate the metrics and review the results
>>> results = ev.evaluate()
>>> print(results)
{'wass-age': 1.7610000000000001}

```

## Further Help

The API is described in the [reference documentation](https://datasciencecampus.github.io/synthgauge/autoapi/index.html).
Please direct any questions to [datacampus@ons.gov.uk](mailto:datacampus@ons.gov.uk).

## Contributing

If you encounter any bugs as part of your usage of `synthgauge`, please file an
issue detailing as much information as possible and include a minimal
reproducible example. If you wish to contribute code to the project, please
refer to the [contribution guidelines](CONTRIBUTING.md).
