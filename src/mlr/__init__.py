"""mlr — personal ML research library.

Importing :mod:`mlr` registers the built-in models and datasets so that
configs can reference them by name.
"""

from mlr.data import synthetic as _synthetic  # noqa: F401  (registers datasets)
from mlr.models.pure import linear_discriminator as _lind  # noqa: F401  (registers models)

__version__ = "0.1.0"
