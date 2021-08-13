from setuptools import setup
import sys

__version__ = "1.1.1"

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

    * Principle Developer and Designer: Brenda Praggastis
    * Visualization: Dustin Arendt, Ji Young Yun
    * High Performance Computing: Tony Liu, Andrew Lumsdaine
    * Principal Investigator: Cliff Joslyn
    * Program Manager: Mark Raugas, Brian Kritzstein
    * Contributors: Sinan Aksoy, Dustin Arendt, Cliff Joslyn, Nicholas Landry, Andrew Lumsdaine, Tony Liu, Brenda Praggastis, Emilie Purvine, Mirah Shi, Francois Theberge

    The code in this repository is intended to support researchers modeling data
    as hypergraphs. We have a growing community of users and contributors.
    Documentation is available at: <https://pnnl.github.io/HyperNetX/>

    For questions and comments contact the developers directly at: <hypernetx@pnnl.gov>

    **New Features of Version 1.0:**

    1. Hypergraph construction can be sped up by reading in all of the data at once. In particular the hypergraph constructor may read a Pandas dataframe object and create edges and nodes based on column headers. The new hypergraphs are given an attribute `static=True`.
    2. A C++ addon called [NWHy](docs/build/nwhy.html) can be used in Linux environments to support optimized hypergraph methods such as s-centrality measures.
    3. A JavaScript addon called [Hypernetx-Widget](docs/build/widget.html) can be used to interactively inspect hypergraphs in a Jupyter Notebook.
    4. Four new tutorials highlighting the s-centrality metrics, static Hypergraphs, [NWHy](docs/build/nwhy.html), and [Hypernetx-Widget](docs/build/widget.html).

    **New Features of Version 1.1**

    1. Static Hypergraph refactored to improve performance across all methods.
    2. Added modules and tutorials for Contagion Modeling, Community Detection, Clustering, and Hypergraph Generation.
    3. Cell weights for incidence matrices may be added to static hypergraphs on construction.
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
