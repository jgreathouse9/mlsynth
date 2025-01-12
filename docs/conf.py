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
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",  # Moved this here directly
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

html_theme = 'press'

# -- Options for EPUB output
epub_show_urls = "footnote"

# -- Sphinx Gallery Configuration
sphinx_gallery_conf = {
    'examples_dirs': './examples',     # Path to your example scripts
    'gallery_dirs': 'auto_examples', # Path where the generated gallery will be stored
    'within_subsection_order': FileNameSortKey,
    'remove_config_comments': True,
    'backreferences_dir': 'generated', # Optional: Link to related documentation
    'doc_module': ('mlsynth',),        # Your main module name
}
