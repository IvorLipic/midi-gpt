import os
from contextlib import contextmanager


@contextmanager
def silence_cpp():
    """Context manager to redirect C-level stdout/stderr to /dev/null."""
    stdout_fd = 1
    stderr_fd = 2
    copy_stdout = os.dup(stdout_fd)
    copy_stderr = os.dup(stderr_fd)

    try:
        with open(os.devnull, 'wb') as devnull:
            os.dup2(devnull.fileno(), stdout_fd)
            os.dup2(devnull.fileno(), stderr_fd)
        yield
    finally:
        os.dup2(copy_stdout, stdout_fd)
        os.dup2(copy_stderr, stderr_fd)
        os.close(copy_stdout)
        os.close(copy_stderr)
