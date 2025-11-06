from .engine.physics import Simulator
from .engine.creatures import Creature

_registry = {
    "crittr": Simulator,
}

def make(name: str, **kwargs):
    if name not in _registry:
        raise KeyError(f"Environment '{name}' not found. Available: {list(_registry.keys())}")
    return _registry[name](**kwargs)

__all__ = ["Simulator", "Creature", "make"]
