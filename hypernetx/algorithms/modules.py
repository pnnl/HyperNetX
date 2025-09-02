import re
import inspect
from functools import wraps


def has_optional_dependency(install_with=None):

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ModuleNotFoundError as e:
                if install_with is None:
                    mod = inspect.getmodule(func)
                    match = re.search(r"hypernetx\.algorithms\.(\w+)\.", mod.__name__)
                    (name,) = match.groups()
                else:
                    name = install_with

                e.add_note(
                    f"You can install optional HypernetX dependencies for this modulule with the following command:\n    >>> pip install hypernetx[{name}]"
                )
                raise e

        return wrapper

    return decorator
