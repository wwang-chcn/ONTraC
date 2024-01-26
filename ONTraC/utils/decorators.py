import inspect
import os
from functools import wraps
from typing import Callable

from ..log import debug


def get_default_args(func) -> dict:
    """Get default arguments of a function.

    Args:
        func (function): A function.

    Returns:
        dict: Default arguments of the function.
    """
    return {
        k: None if v.default is inspect.Parameter.empty else v.default
        for k, v in inspect.signature(func).parameters.items()
    }


def selective_args_decorator(func) -> Callable:
    """Decorator that allows a function to accept only a subset of its arguments.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # debug(f'selective_args_decorator: args: {args}')
        # debug(f'selective_args_decorator: kwargs: {list(kwargs.keys())}')
        default_args = {}
        for key, value in inspect.signature(func).parameters.items():
            # debug(f'key: {key}, value: {value}')
            if value.default is inspect.Parameter.empty:  # without default value
                if key in ['args', 'kwargs', 'self']:  # skip
                    continue
                elif key in kwargs:  # with input value
                    default_args[key] = kwargs[key]
                else:  # without input value
                    raise TypeError(f'{func.__name__}() missing 1 required positional argument: {key}')
            else:  # with default value
                default_args[key] = kwargs[key] if key in kwargs else value.default
        if 'kwargs' in default_args:
            default_args.update(kwargs)
        # debug(f'default_args: {list(default_args.keys())}')

        # call the function with the updated arguments
        return func(*args, **default_args)

    return wrapper


def epoch_filter_decorator(func: Callable) -> Callable:
    """
    Epoch filter decorator
    :param func: function
    :return: function
    """

    @wraps(func)
    def wrapper(*args, epoch_filter: Callable, **kwargs) -> None:  # add epoch_filter argument
        # debug(f'epoch_filter_decorator: args: {args}')
        # debug(f'epoch_filter_decorator: kwargs: {list(kwargs.keys())}')
        epoch: int = kwargs.get('epoch')  # type: ignore
        output_dir: str = kwargs.get('output_dir')  # type: ignore

        # Check the epoch_filter condition
        if epoch_filter(epoch):
            # Check if output_dir of this epoch exists
            epoch_output_dir: str = os.path.join(output_dir, f'Epoch_{epoch}')
            if not os.path.exists(epoch_output_dir):
                os.mkdir(epoch_output_dir)
            # Call the original function if condition is True
            return func(*args, **kwargs)
        else:
            # Otherwise, return None or some default value
            return None

    # Update the function signature
    sig = inspect.signature(func)
    parms = list(sig.parameters.values())
    parms.append(
        inspect.Parameter(name='epoch_filter',
                          kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          default=inspect.Parameter.empty,
                          annotation=Callable))
    wrapper.__signature__ = sig.replace(parameters=parms)  # type: ignore

    return wrapper