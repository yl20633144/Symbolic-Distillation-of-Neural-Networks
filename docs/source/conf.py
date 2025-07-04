# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Symbolic Distillation'
copyright = '2025, Yihao Liu'
author = 'Yihao Liu'
release = '1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'autoapi.extension',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # if using Google/Numpy-style docstrings
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))


html_theme = 'alabaster'
html_static_path = ['_static']
# -- AutoAPI configuration ---------------------------------------------------
autoapi_type               = 'python'
autoapi_dirs = ['../../src']
# scan only these two files at the project root:
autoapi_file_patterns      = ['*.py']
# tell AutoAPI where to write its rst
autoapi_root              = 'api'
# automatically add the API index into toctree
autoapi_add_toctree_entry = True
# include documented and undocumented members
autoapi_options           = ['members', 'undoc-members', 'show-inheritance']
