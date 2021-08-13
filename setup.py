from setuptools import setup
import sys

__version__ = "1.1"

if sys.version_info < (3, 7):
    sys.exit("HyperNetX requires Python 3.7 or later.")

setup(
    name="hypernetx",
    packages=[
        "hypernetx",
        "hypernetx.algorithms",
        "hypernetx.algorithms.contagion",
        "hypernetx.classes",
        "hypernetx.drawing",
        "hypernetx.reports",
        "hypernetx.utils",
        "hypernetx.utils.toys",
    ],
    version=__version__,
    author="Brenda Praggastis, Dustin Arendt, Sinan Aksoy, Emilie Purvine, Cliff Joslyn, Nicholas Landry",
    author_email="hypernetx@pnnl.gov",
    url="https://github.com/pnnl/HyperNetX",
    description="HyperNetX is a Python library for the creation and study of hypergraphs.",
    install_requires=[
        "networkx>=2.2,<3.0",
        "numpy>=1.15.0,<2.0",
        "scipy>=1.1.0,<2.0",
        "matplotlib>3.0",
        "scikit-learn>=0.20.0",
        "pandas>=0.23",
        "celluloid>=0.2.0",
    ],
    license="3-Clause BSD license",
    long_description="""
    The HyperNetX library provides classes and methods for the analysis
    and visualization of complex network data modeled as hypergraphs. 
    The library generalizes traditional graph metrics.

    HypernetX was developed by the Pacific Northwest National Laboratory for the
    Hypernets project as part of its High Performance Data Analytics (HPDA) program.
    PNNL is operated by Battelle Memorial Institute under Contract DE-ACO5-76RL01830.
    """,
    extras_require={
        "testing": ["pytest>=4.0"],
        "tutorials": ["jupyter>=1.0"],
        "documentation": ["sphinx>=1.8.2", "nb2plots>=0.6", "sphinx-rtd-theme>=0.4.2"],
        "all": [
            "sphinx>=1.8.2",
            "nb2plots>=0.6",
            "sphinx-rtd-theme>=0.4.2",
            "pytest>=4.0",
            "jupyter>=1.0",
        ],
    },
)
