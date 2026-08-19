"""``python -m mozo`` -- the same commands as the ``mozo`` script.

    python -m mozo start
    python -m mozo version
"""

from mozo.cli import cli

if __name__ == "__main__":
    cli()
