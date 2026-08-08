"""mlr — personal ML research library.

Importing :mod:`mlr` registers the built-in models and datasets so that
configs can reference them by name.
"""

from mlr.data import synthetic as _synthetic  # noqa: F401  (registers datasets)
from mlr.models.pure import linear_discriminator as _lind  # noqa: F401  (registers models)
from mlr.models.pure import mle as _mle  # noqa: F401  (registers models)
from mlr.models.torch import HAS_TORCH as _HAS_TORCH

if _HAS_TORCH:
    from mlr.models.torch import linear_discriminator as _lind_torch  # noqa: F401
    from mlr.models.torch import mle as _mle_torch  # noqa: F401

__version__ = "0.1.0"
