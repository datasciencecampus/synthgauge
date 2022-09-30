# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import synthgauge as sg

# -- Project information -----------------------------------------------------

project = "SynthGauge"
copyright = "2022, Data Science Campus"
author = "Ali Cass, Michaela Lawrence, Tom White and Henry Wilde"

# The full version, including alpha/beta/rc tags
release = sg.__version__
napoleon_include_private_with_doc = False

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "autoapi.extension",
    "nbsphinx",
    "nbsphinx_link",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# sphinx-autoapi configuration
autoapi_dirs = ["../../src"]
autoapi_options = [
    "members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Logo for sidebar
html_logo = "./_static/favicon.png"

# Faviocon
html_favicon = "./_static/favicon.png"

# # replace "view page source" with "edit on github" in Read The Docs theme
# #  * https://github.com/readthedocs/sphinx_rtd_theme/issues/529
# html_context = {'display_github': True,
#                 'github_user': 'datasciencecampus',
#                 'github_repo': 'synthgauge',
#                 'github_version': 'main/',
#                 }
