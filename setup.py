from setuptools import setup
import sys

__version__ = "1.0"

if sys.version_info < (3, 7):
    sys.exit("HyperNetX requires Python 3.7 or later.")

setup(
    name="hypernetx",
    packages=[
        "hypernetx",
        "hypernetx.algorithms",
        "hypernetx.classes",
        "hypernetx.drawing",
        "hypernetx.utils",
    ],
    version=__version__,
    author="Brenda Praggastis, Dustin Arendt, Emilie Purvine, Cliff Joslyn",
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
    ],
    license="3-Clause BSD license",
    long_description="""
    The HyperNetX library provides classes and methods for the analysis
    and visualization of complex network data modeled as hypergraphs. 
    The library generalizes traditional graph metrics.

    HypernetX was developed by the Pacific Northwest National Laboratory for the
    Hypernets project as part of its High Performance Data Analytics (HPDA) program.
    PNNL is operated by Battelle Memorial Institute under Contract DE-ACO5-76RL01830.

    * Principle Developer and Designer: Brenda Praggastis
    * Visualization: Dustin Arendt, Ji Young Yun
    * High Performance Computing: Tony Liu, Andrew Lumsdaine
    * Principal Investigator: Cliff Joslyn
    * Program Manager: Mark Raugas, Brian Kritzstein
    * Mathematics, methods, and algorithms: Sinan Aksoy, Dustin Arendt, Cliff Joslyn, Andrew Lumsdaine, Tony Liu, Brenda Praggastis, and Emilie Purvine

    The code in this repository is intended to support researchers modeling data
    as hypergraphs. We have a growing community of users and contributors.
    Documentation is available at: <https://pnnl.github.io/HyperNetX/>

    For questions and comments contact the developers directly at:
        <hypernetx@pnnl.gov>
    """,
    extras_require={
        "testing": ["pytest>=4.0"],
        "notebooks": [
            "jupyter>=1.0",
        ],
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
