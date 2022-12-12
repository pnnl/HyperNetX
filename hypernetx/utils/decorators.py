import warnings
from functools import wraps

from decorator import decorator

import hypernetx as hnx
from hypernetx.exception import NWHY_WARNING

__all__ = [
    "not_implemented_for",
    "warn_nwhy",
]


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


def warn_nwhy(func):
    """Decorator for methods that allow the deprecated `use_nwhy` kwarg

    As of HyperNetX v2.0.0, NWHy C++ backend is no longer supported.
    Public references to the deprecated NWHy add-on will be removed from the Hypergraph
    API in a future release.

    Warns
    -----
    FutureWarning
        If kwargs contain ``use_nwhy=True``
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if kwargs.get("use_nwhy"):
            kwargs.update(use_nwhy=False)
            warnings.simplefilter("always", FutureWarning)
            warnings.warn(NWHY_WARNING, FutureWarning, stacklevel=2)
            warnings.simplefilter("default", FutureWarning)

        return func(*args, **kwargs)

    return wrapper
