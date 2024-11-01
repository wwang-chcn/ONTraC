import os
import shutil
from contextlib import contextmanager
from optparse import Values


@contextmanager
def temp_dirs(options: Values):
    """
    Create temporary directories for testing.
    :param options: Values, options.
    :return: None.
    """
    try:
        # Create temporary directories
        if hasattr(options, 'NN_dir'):
            os.makedirs(options.NN_dir, exist_ok=True)
        if hasattr(options, 'GNN_dir'):
            os.makedirs(options.GNN_dir, exist_ok=True)
        if hasattr(options, 'NT_dir'):
            os.makedirs(options.NT_dir, exist_ok=True)
        if hasattr(options, 'output'):
            os.makedirs(options.output, exist_ok=True)

        # Yield to the test
        yield

    finally:
        # Remove temporary directories
        if hasattr(options, 'NN_dir'):
            shutil.rmtree(options.NN_dir)
        if hasattr(options, 'GNN_dir'):
            shutil.rmtree(options.GNN_dir)
        if hasattr(options, 'NT_dir'):
            shutil.rmtree(options.NT_dir)
        if hasattr(options, 'output'):
            shutil.rmtree(options.output)
