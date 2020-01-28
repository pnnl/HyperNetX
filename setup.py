from setuptools import setup 
import sys, os.path

__version__ = '0.2.7'

if sys.version_info < (3,6):
    sys.exit('HyperNetX requires Python 3.6 or later.')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'hypernetx'))

setup(
    name='hypernetx',
    packages=['hypernetx',
              'hypernetx.classes',
              'hypernetx.drawing',
              'hypernetx.reports'],
    version=__version__,
    author="Brenda Praggastis, Dustin Arendt, Emily Purvine, Cliff Joslyn",
    author_email="hypernetx@pnnl.gov",
    url='https://github.com/pnnl/HyperNetX',
    description='HyperNetX is a Python library for the creation and study of hypergraphs.',
    install_requires=['networkx>=2.2,<3.0','numpy>=1.15.0,<2.0','scipy>=1.1.0,<2.0','matplotlib>3.0','scikit-learn>=0.20.0'],
    license='3-Clause BSD license',
    long_description='''
    The HyperNetX library provides classes and methods for complex network data. 
    HyperNetX uses data structures designed to represent set systems containing 
    nested data and/or multi-way relationships. The library generalizes traditional 
    graph metrics to hypergraphs. 

    The current version is preliminary. We are actively testing and would be grateful 
    for comments and suggestions. Expect changes in both class names and methods as 
    many of the requirements demanded of the library are worked out.
    ''',
    extras_require={
        'testing':['pytest>=4.0'],
        'notebooks':['jupyter>=1.0','pandas>=0.23'],
        'tutorials':['jupyter>=1.0','pandas>=0.23'],
        'documentation':['sphinx>=1.8.2','nb2plots>=0.6','sphinx-rtd-theme>=0.4.2'],
        'all':['sphinx>=1.8.2','nb2plots>=0.6','sphinx-rtd-theme>=0.4.2','pytest>=4.0','jupyter>=1.0','pandas>=0.23']
    }
)

### Since this package is still in development, please install in a virtualenv or conda environment.
### See README for installations instructions


