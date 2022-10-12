# v2.2.0 - 2022-10-12

This release adds two new features to the `metrics.propensity` module:

- An additional method for estimating pMSE null statistics that only uses the
  real data. Eventually, this will become the default behaviour and a warning
  is issued if the old permutation-based method is used. Details of the method
  can be found in [Bowen and Snoke (2021)](https://doi.org/10.29012/jpc.748).
- The SPECKS metric: a propensity-score-based metric that uses the KS distance.
  Details in [Bowen et al. (2021)](https://doi.org/10.1007/s40300-021-00201-0).

# v2.1.0 - 2022-09-30

In this minor release, we add a new set of utility metrics from the 2018 NNIST
competition. We also fix a couple of small bugs and make some minor improvements
to the documentation.

## Features

- Add generic utility metrics used in the 2018 NIST competition - *k*-way
  marginal comparison and higher-order conjunctions. These are stored in the
  `synthgauge.metrics.nist` module.
- Implement doctesting across README and docstrings.
- Use `pytest-randomly` to make test suite more robust.

## Documentation

- Rewrite the package README and the univariate metric docstrings to be clearer
  and more readily tested. Also, the README is now version-invariant and
  reflects the fact `synthgauge` is on PyPI.
- Update the contribution documentation, including fixing a couple of typos.
- Hide "hidden" functions from the API reference.

## Bug fixes

- Workaround for a strange seaborn@0.11.2 bug
- Streamline dependencies (everything is now inside `setup.cfg`)

# v2.0.0 - 2022-07-19

This release hopes to improve the reproducibility of `synthgauge` by adding
reliable random number sampling, robust tests, fuller documentation, code
formatting, continuous integration, code styling and bug fixes.

As well as these additions, several parts of the codebase have been refactored
and reorganised to be clearer to users and developers at the expense of
breaking back-compatibility.

Finally, the licence under which this software is released has been updated to
the MIT License.

## Features

- Pseudo-random number seeding is now carried out according to best practices,
  allowing for complete reproducibility with the implemented metrics.
- Full property-based testing suite with 100% coverage from `pytest`,
  `hypothesis` and `pytest-cov`.
- Code stylers and GitHub Action CI workflow via `black`, `flake8`,
  `interrogate`, `isort`, `tox`.
- Single-source version number within the source code.
- Most meant-to-be-private functions now named with a leading underscore.
- Correlation MSD can now use Spearman's method.
- More control over outlier detection when using minimum nearest-neighbour
  privacy metric by exposing its parameter in the metric call.
- Expose colour map in `plot.plot_crosstab`.
- Streamline number of warnings following more explicit documentation.

## Deprecations and removals

- Many parameter and function names have been changed to align with best
  practices for Python as well as to be consistent and concise.
- Feature density difference functions have been moved to their own module:
  `metrics.density`.
- All univariate metrics (distances, divergences and hypothesis tests) are now
  in their own module: `metrics.univariate`.
- Module containing only the `Evaluator` class now called `evaluator`.
- Correlation MSD metrics combined into single metric with `method` argument.
- Ability to pass single column where a list is typical no longer allowed.
- Categorical columns cannot be used to make Q-Q plots anymore.
- Remove all previous tests except the metric examples, which act as regression
  tests.

## Bug fixes

- Catch random number leaking when applying the logistic propensity model.
- Base categorical encoding on combined data. Previously, users would get
  inconsistent encoding when the real and synthetic features did not have
  identical category sets.
- Explicitly set k-means clustering to old algorithm following change in the
  default in `scikit-learn`.
- Refactor and modularise several larger functions (particularly metrics).
- Calculate correlation MSDs using the upper triangular correlation matrix to
  avoid effect of double-counting.
- Fix error thrown in `metrics.privacy.sample_overlap_score` if using the whole
  sample and synthetic data shorter than the real data.
- Use `k - 1` rather than `k` in propensity null case statistics.
- Remove unnecessary `if __name__ == "__main__": pass` blocks.
- Address `FutureWarning` from `pandas` for use of `pandas.DataFrame.append`.
- Remove

## Documentation

- Richer hosted documentation (example notebook and welcome page).
- Clearer contribution guidelines.
- Full and corrected docstrings for all modules, classes and functions.

# v1.0.0 - 2022-02-09

Initial release of `synthgauge`.
