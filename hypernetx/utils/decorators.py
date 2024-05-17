import warnings
from functools import wraps

from decorator import decorator

import hypernetx as hnx


__all__ = ["not_implemented_for", "warn_to_be_deprecated"]


def not_implemented_for(*object_types):
    """Decorator to hypergraph methods to mark algorithms as not implemented
    Ruthlessly copied from NetworkX.

    Parameters
    ----------
    object_types : container of strings
        Entries must be one of ['static','dynamic']

    Returns
    -------
    _require : function
        The decorated function.

    Raises
    ------
    HyperNetXNotImplemented
    If any of the packages cannot be imported

    Notes
    -----
    Multiple types are joined logically with "and".
    For "or" use multiple @not_implemented_for() lines.

    Examples
    --------
    Decorate functions like this::

       @not_implemented_for_hypergraph('static')
       def add_edges_from(H):
           pass

    """

    @decorator
    def _not_implemented_for(not_implemented_for_func, *args, **kwargs):
        this_object = args[0]
        terms = {"static": this_object.isstatic, "dynamic": not this_object.isstatic}
        match = True
        try:
            for t in object_types:
                match = match and terms[t]
        except KeyError:
            raise KeyError("use one of [static, dynamic]")
        if match:
            msg = f'{not_implemented_for_func.__name__} is not implemented for {" ".join(object_types)} {this_object.__class__.__name__}'
            raise hnx.HyperNetXNotImplementedError(msg)
        else:
            return not_implemented_for_func(*args, **kwargs)

    return _not_implemented_for


def warn_to_be_deprecated(func):
    """Decorator for methods that are to be deprecated

    Public references to deprecated methods or functions will be removed from the Hypergraph API in a future release.

    Warns
    -----
    FutureWarning
    """

    deprecation_warning_msg = (
        "This method or function will be deprecated in a future release. "
        "Public references to this method or function will be removed from the "
        "Hypergraph API in a future release."
    )

    @wraps(func)
    def wrapper(*args, **kwargs):
        warnings.simplefilter("always", FutureWarning)
        warnings.warn(deprecation_warning_msg, FutureWarning, stacklevel=2)
        warnings.simplefilter("default", FutureWarning)
        return func(*args, **kwargs)

    return wrapper
