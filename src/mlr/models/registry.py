"""Model registry.

Models register themselves by name so experiment configs can reference them:

    @register_model("linear-discriminator")
    class LinearDiscriminator: ...

A model class must accept its hyperparameters as keyword arguments to
``__init__`` and expose ``fit(X, y, on_epoch=None)`` and ``predict(X)``.
"""

_MODELS: dict[str, type] = {}


def register_model(name: str):
    def decorator(cls: type) -> type:
        if name in _MODELS:
            raise ValueError(f"model {name!r} is already registered")
        _MODELS[name] = cls
        return cls

    return decorator


def get_model(name: str) -> type:
    try:
        return _MODELS[name]
    except KeyError:
        raise KeyError(
            f"unknown model {name!r}; registered models: {sorted(_MODELS)}"
        ) from None


def list_models() -> list[str]:
    return sorted(_MODELS)
