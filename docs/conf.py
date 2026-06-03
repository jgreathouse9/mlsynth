import os
import sys
from sphinx_gallery.sorting import FileNameSortKey  # Import FileNameSortKey

sys.path.insert(0, os.path.abspath("../"))

# Configuration file for the Sphinx documentation builder.

# -- Project information

project = "mlsynth"
copyright = "2025, Jared Greathouse"
author = "Jared Greathouse"

release = "0.1"
version = "0.1.0"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton"
]

# Napoleon settings.
#
# The estimator helper modules define dozens of frozen dataclasses (BVSSInputs,
# SSCInputs, MAREXConfig, ...) whose docstrings document shape-typed attributes
# and parameters such as ``Y_pre : numpy.ndarray, shape (T0, m)``. With the
# default ``napoleon_use_param``/``napoleon_use_rtype`` (True), Napoleon renders
# the type portion of each Parameters/Returns entry as a typed Python-domain
# field, and the domain then tries to turn the bare identifier tokens in those
# type strings (T, N, TN, n, T0, m, ...) into cross-references. Because every
# estimator's Inputs dataclass exposes identically-named attributes, those bare
# names resolve to many targets and Sphinx emits ~82 "more than one target
# found for cross-reference 'T'/'N'/'n'/..." warnings.
#
# Rendering parameters/returns as definition lists instead of typed domain
# fields (the two settings below) stops Napoleon from emitting those auto
# cross-referenced type fields, driving the ambiguous-attribute xref warnings to
# zero, without suppressing any warning category. This is preferred over the
# blunt ``suppress_warnings = ["ref.python"]``, which would also hide genuine
# broken Python references.
napoleon_use_param = False
napoleon_use_rtype = False

latex_elements = {"preamble": r"\usepackage{mathtools}"}

def setup(app):
    app.add_js_file("static/custom_mathjax.js")

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = 'sphinx_book_theme'

# -- Options for EPUB output
epub_show_urls = "footnote"
