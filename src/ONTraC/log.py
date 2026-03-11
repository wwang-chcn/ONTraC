"""This module accounts for logging functionality in ONTraC, including functions for writing messages to stdout and stderr with timestamps and different log levels (debug, info, warning, error, critical)."""

import platform
import sys
import time


if platform.system() == 'Windows':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8') # type: ignore
    else:
        sys.stdout.encoding = 'utf-8' # type: ignore
    if sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def get_current_time() -> str:
    """Get current time.
    
    Returns
    -------
    str, current time."""
    return time.strftime('%H:%M:%S', time.localtime())


def write_direct_message(message: str) -> None:
    """Write direct message to stdout.
    
    Parameters
    ----------
    message :
        str, message.
    
    Returns
    -------
    None."""
    curr_time_str = get_current_time()
    sys.stdout.write(f'{curr_time_str} --- {message}\n')
    sys.stdout.flush()


def debug(message: str) -> None:
    """Debug message.
    
    Parameters
    ----------
    message :
        str, message.
    
    Returns
    -------
    None."""
    write_direct_message(f'DEBUG: {message}')


def info(message: str) -> None:
    """Info message.
    
    Parameters
    ----------
    message :
        str, message.
    
    Returns
    -------
    None."""
    write_direct_message(f'INFO: {message}')


def write_direct_message_err(message: str) -> None:
    """Write direct message to stderr.
    
    Parameters
    ----------
    message :
        str, message.
    
    Returns
    -------
    None."""
    curr_time_str = get_current_time()
    sys.stderr.write(f'{curr_time_str} --- {message}\n')
    sys.stderr.flush()


def warning(message: str) -> None:
    """Warning message.
    
    Parameters
    ----------
    message :
        str, message.
    
    Returns
    -------
    None."""
    write_direct_message_err(f'WARNING: {message}')


def error(message: str) -> None:
    """Error message.
    
    Parameters
    ----------
    message :
        str, message.
    
    Returns
    -------
    None."""
    write_direct_message_err(f'ERROR: {message}')


def critical(message: str) -> None:
    """Critical message.
    
    Parameters
    ----------
    message :
        str, message.
    
    Returns
    -------
    None."""
    write_direct_message_err(f'CRITICAL: {message}')


__all__ = ['debug', 'info', 'warning', 'error', 'critical']
