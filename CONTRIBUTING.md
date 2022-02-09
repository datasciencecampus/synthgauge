<p align="center">
  <img src="images/logo.png" />
</p>

First off, thank you for considering contributing to `synthgauge`. It's people like you that make `synthgauge` such a great tool.

Following these guidelines helps to communicate that you respect the time of the developers managing and developing this open source project. In return, they should reciprocate that respect in addressing your issue, assessing changes, and helping you finalize your pull requests.

<a id="what-to-contribute"> </a>

## What to Contribute

`synthgauge` is an open source project and we love to receive contributions from our community — you! There are many ways to contribute, from writing tutorials or blog posts, improving the documentation, submitting bug reports and feature requests or writing code which can be incorporated into `synthgauge` itself.

### Code Contributions
To contribute to `synthgauge` users should fork the repository and checkout a new branch. Any commits on this branch will then have to pass the pre-commit hooks and, where there are failures, these will be flagged for remedy. Likewise, prior to submitting any contributions, users must ensure that their code passes all the tests provided in the test suite. To submit a contribution users must make a pull request of their changes and await review by the development team.

Generally, new contributions will fall into two categories: metrics and visualisations. Such contributions should be placed in `synthgauge.metrics` and `synthgauge.plot` respectively.

As the metrics are stored in a subpackage contributers should store any metrics code in
the appropriate modules. Currently these are `classification`, `cluster`, `correlation`, `privacy`, `propensity` and `univariate distance`.

If a new metric cannot be categorised according to the current schemes contributers can create their own module. The suitablity of this new module will be assessed by the development team during any subsequent pull request.

Other contributions can include helpful utility functions or toy data generation and these should be placed in `synthgauge.utils` and `synthgauge.datasets` respectively.

Finally, contributions must be documented using the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) format. Corresponding documentation can then be generated using `Sphinx`.

<a id="ground-rules"> </a>

## Ground Rules

### Responsibilities
* Ensure cross-platform compatibility for every change that's accepted. Windows, Mac, Debian & Ubuntu Linux.
* Create issues for any major changes and enhancements that you wish to make. Discuss things transparently and get community feedback.
* Don't add any classes to the codebase unless absolutely needed. Err on the side of using functions.
* Keep feature versions as small as possible, preferably one new feature per version.
* Be welcoming to newcomers and encourage diverse new contributors from all backgrounds. See the [Python Community Code of Conduct](https://www.python.org/psf/codeofconduct/).

Working on your first Pull Request? You can learn how from this *free* series, [How to Contribute to an Open Source Project on GitHub](https://egghead.io/courses/how-to-contribute-to-an-open-source-project-on-github).

<a id="getting-started"> </a>

## Getting started

### How to submit a Contribution.

1. Create your own fork of the code
2. Install the requirements listed in `requirements.dev.txt`
3. Install the pre-commit hooks
4. Do the changes in your fork
5. If you like the change and think the project could use it:
    * Be sure you have followed the code style for the project
    * Be sure your code passes all the tests
    * Generate/update the API Reference
    * Send a pull request

## How to report a bug

If you find a security vulnerability, do NOT open an issue. Email datacampus@ons.gov.uk instead.

When filing an issue, make sure to answer the questions in the Bug template.

## Code review process

The Campus looks at Pull Requests on a regular basis but cannot unfortunately guarantee prompt implementation of Pull Requests.
