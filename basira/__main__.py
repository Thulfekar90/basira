"""``python -m basira`` is the same as the ``basira`` command."""
import sys

from .cli import main

# The guard matters: ``basira scan --jobs N`` starts worker processes that import this
# module again (as ``__mp_main__``); without it every worker would re-run the command.
if __name__ == "__main__":
    sys.exit(main())
