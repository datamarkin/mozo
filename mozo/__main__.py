"""
Entry point for running `python -m mozo`.

This delegates to the CLI start command for consistency.
"""

from mozo.cli import cli
import sys


def main():
    """
    Entry point when running `python -m mozo`.
    Defaults to the start command for backward compatibility.
    """
    # If no arguments, default to 'start --reload'
    if len(sys.argv) == 1:
        sys.argv.extend(['start', '--reload'])

    cli()


if __name__ == "__main__":
    main()
