# Copyright © 2018 Battelle Memorial Institute
# All rights reserved.

"""
Base classes for HyperNetX exceptions
"""

NWHY_WARNING = (
    "As of HyperNetX v2.0.0, NWHy C++ backend is no longer supported. "
    "Public references to the deprecated NWHy add-on will be removed from the "
    "Hypergraph API in a future release."
)


class HyperNetXException(Exception):
    """Base class for exceptions in HyperNetX."""


class HyperNetXError(HyperNetXException):
    """Exception for a serious error in HyperNetX"""


class HyperNetXNotImplementedError(HyperNetXError):
    """Exception for methods not implemented for an object type."""
