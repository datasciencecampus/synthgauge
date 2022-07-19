# Contribution Guidelines

First off, thank you for considering contributing to `synthgauge`. It's people
like you that make `synthgauge` such a great tool.

There are many ways to contribute, from writing tutorials or blog posts,
improving the documentation, submitting bug reports and feature requests, or
writing code which can be incorporated into `synthgauge` itself.

Following these guidelines helps to communicate that you respect the time of
the developers managing and developing this open source project. In return,
they should reciprocate that respect in addressing your issue, assessing
changes, and helping you finalise your pull requests.

## Code Contributions

Generally, code contributions will fall into two categories: metrics and
visualisations. Such contributions should be placed in `synthgauge.metrics` and
`synthgauge.plot`, respectively.

As the metrics are stored in a subpackage, contributors should store any new
metric code in the appropriate module. Currently, these are `classification`,
`cluster`, `correlation`, `density`, `privacy`, `propensity` and `univariate`.

If a new metric cannot be categorised according to the current schemes
contributers can create their own module. The suitablity of this new module
will be assessed by the development team during any subsequent pull request.

Other contributions can include helpful utility functions or toy data
generation and these should be placed in `synthgauge.utils` and
`synthgauge.datasets`, respectively.

### Installation

To contribute to `synthgauge`, you should fork the repository, clone it
locally, and checkout a new branch. You should install the library as editable
as well:

```bash
$ cd /path/to/synthgauge
$ python -m pip install -e .
```

We use several tools to ensure the reproducibility and consistency of the
`synthgauge` codebase, and your code will have to adhere to these principles as
well. To install these tools along with the other development requirements, run
the following:

```bash
$ python -m pip install -r requirements.dev.txt
```

### Testing

Within `synthgauge`, we make use of property-based and example regression tests
to ensure our code works as it should. Any new functions you implement must
have tests to accompany them. We use `pytest`, `hypothesis` and `coverage` to
run our testing suit. You can find examples of tests in their documentation and
in the `tests` directory of the repository.

Please ensure that all tests pass and that you have 100% coverage before you
open a pull request:

```bash
$ python -m pytest tests --cov=synthgauge
```

### Code style

We use pre-commit hooks to help keep our codebase clean. Please install
`pre-commit` in your local repository before you start developing your
contributions. Any issues will be flagged to you for remedying before you
commit your code.

As well as this, we use three formatting tools to ensure a consistent code
style: `black`, `isort` and `flake8`. Please ensure that you run these tools
against the core codebase before submitting your contribution for review. These
tools are configured for you, so you need only run the following commands:

```bash
$ python -m black src tests
$ python -m isort src tests
$ python -m flake8 src tests
```

### Documentation

All contributions must be documented using the
[numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) format.

The `synthgauge` documentation is hosted on GitHub Pages, and an important part
of that is the API reference material. So, any new additions to the API
(functions, modules, etc.) will require updated reference material. We use
`sphinx` for our documentation. To update the reference material, first delete
everything in the `docs` directory except the empty `.nojekyll` file. This file
ensures that the documentation renders properly online. Now, render new copies
of the documentation like so:

```bash
$ cd /path/to/synthgauge/docs_src
$ sphinx-build -a -E source ../docs
```

You can preview the documentation by opening the file
`/path/to/synthgauge/docs/index.html` in your browser. Once you're happy with
the documentation, commit your changes.

### Submission

Now you've written and documented your code, and passed the checks, you're
ready to submit your contribution. To do so, you must make a pull request with
your changes to the original (upstream) repository and await review by the
development team.

The Campus looks at pull requests on a regular basis but cannot, unfortunately,
guarantee a prompt implementation of them.

Working on your first pull request? You can learn how from this *free* series,
[How to Contribute to an Open Source Project on
GitHub](https://egghead.io/courses/how-to-contribute-to-an-open-source-project-on-github).

### Responsibilities

* Ensure cross-platform compatibility for every change that's accepted.
  Windows, Mac, Debian & Ubuntu Linux. The current CI workflow checks against
  Windows and Ubuntu.
* Create issues for any major changes and enhancements that you wish to make.
  Discuss things transparently and get community feedback.
* Don't add any classes to the codebase unless absolutely needed. Err on the
  side of using functions.
* Keep feature versions as small as possible, preferably one new feature per
  version.
* Be welcoming to newcomers and encourage diverse new contributors from all
  backgrounds. See the [Python Community Code of
  Conduct](https://www.python.org/psf/codeofconduct/).

## Bug reports

When reporting a bug, please provide as much detail as you can about the issue,
including a minimal working example, where appropriate. Like other
contributions, the development team review bug reports regularly and will aim
to patch them as soon as possible. However, we cannot guarantee these changes
will be prompt. If you comfortable doing so, consider making a code
contribution addressing the issue.

If you find a security vulnerability, **do not open an issue**. Email
[datacampus@ons.gov.uk](mailto:datacampus@ons.gov.uk) instead.
