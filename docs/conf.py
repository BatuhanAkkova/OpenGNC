# Configuration file for the Sphinx documentation builder.

from pathlib import Path
import sys

sys.path.insert(0, str((Path(__file__).resolve().parent.parent / "src").resolve()))

project = "OpenGNC"
author = "Batuhan Akkova"

extensions = [
    "myst_nb",
    "jupyter_sphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".md": "markdown",
}

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "html_image",
    "colon_fence",
    "linkify",
]

nb_execution_mode = "off"
myst_dmath_double_inline = True
myst_update_mathjax = False

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
mathjax_config = {
    "tex2jax": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "processEscapes": True,
    },
}

master_doc = "index"
templates_path = ["_templates"]
html_theme = "sphinx_rtd_theme"
(Path(__file__).resolve().parent / "_static").mkdir(exist_ok=True)
html_static_path = ["_static"]
suppress_warnings = [
    "mystnb.unknown_mime_type",
    "ref.python",
]
