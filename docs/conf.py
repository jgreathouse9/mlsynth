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

# -- Sphinx Gallery Configuration
sphinx_gallery_conf = {
    'examples_dirs': ['auto_examples'],   # Path to your examples folder
    'gallery_dirs': ['auto_examples'],  # Path to the gallery build output
    'filename_pattern': r'auto_examples/.*\.py'
}

