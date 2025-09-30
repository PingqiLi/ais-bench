import sys
from datetime import datetime


def is_running_in_background():
    # check whether stdin and stdout are connected to TTY
    stdin_is_tty = sys.stdin.isatty()
    stdout_is_tty = sys.stdout.isatty()

    # if stdin and stdout are not connected to TTY, the script is running in background
    return not (stdin_is_tty and stdout_is_tty)


def get_current_time_str():
    return datetime.now().strftime('%Y%m%d_%H%M%S')