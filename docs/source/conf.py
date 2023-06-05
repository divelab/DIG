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
import sys
import torch_geometric
import sphinx_rtd_theme

sys.path.insert(0,'..')
sys.path.insert(0,'../..')

import dig.auggraph.dataset
import dig.auggraph.method
import dig.sslgraph.dataset
import dig.sslgraph.method
import dig.sslgraph.utils
import dig.sslgraph.evaluation
import dig.ggraph.dataset
import dig.ggraph.method
import dig.ggraph.utils
import dig.ggraph.evaluation
import dig.ggraph3D.dataset
import dig.ggraph3D.method
import dig.ggraph3D.utils
import dig.ggraph3D.evaluation
import dig.xgraph.dataset
import dig.xgraph.method
import dig.xgraph.evaluation
import dig.threedgraph.dataset
import dig.threedgraph.method
import dig.threedgraph.utils
import dig.threedgraph.evaluation
import dig.fairgraph.dataset
import dig.fairgraph.method

# -- Project information -----------------------------------------------------

project = 'DIG: Dive into Graphs'
copyright = '2021, DIVE Lab'
author = 'DIVE Lab'

# The full version, including alpha/beta/rc tags
release = '1.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'autodocsumm',
]


# Add any paths that contain templates here, relative to this directory.

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
# html_theme = "furo"

intersphinx_mapping = {'python': ('https://docs.python.org/', None),
                       'torch': ('https://pytorch.org/docs/master/', None),
                       'torch_geometric': ('https://pytorch-geometric.readthedocs.io/en/latest/', None)}

add_module_names = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

autodoc_default_options = {'autosummary-no-titles': True,
                           'autosummary-force-inline': True,
                           'autosummary-nosignatures': True,
                          }


def setup(app):
    def skip(app, what, name, obj, skip, options):
        members = [
            '__init__',
            '__repr__',
            '__weakref__',
            '__dict__',
            '__module__',
            'qed',
        ]
        return True if name in members else skip

    app.connect('autodoc-skip-member', skip)
