# Copyright © 2018 Battelle Memorial Institute
# All rights reserved.

'''
Base classes for HyperNetX exceptions
'''


class HyperNetXException(Exception):
    """Base class for exceptions in HyperNetX."""


class HyperNetXError(HyperNetXException):
    """Exception for a serious error in HyperNetX"""


class HyperNetXNotImplementedError(HyperNetXError):
    """Exception for methods not implemented for an object type."""
