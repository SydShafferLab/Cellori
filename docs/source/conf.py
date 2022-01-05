import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'Cellori'
copyright = '2022, William Niu'
author = 'William Niu'

release = '3.0'
version = '3.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = [
    'styles.css',
]

# -- Options for EPUB output
epub_show_urls = 'footnote'

# -- Extension configuration

autodoc_member_order = 'bysource'
autodoc_mock_imports = [
    'matplotlib',
    'numba',
    'numpy',
    'cv2',
    'pyside6',
    'scipy',
    'SimpleITK',
    'skimage',
    'stitchwell',
    'tifffile'
]
