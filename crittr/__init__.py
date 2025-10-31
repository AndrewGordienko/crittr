from .physics import Simulator
from .creatures import Creature

# --- simple registry ---
_registry = {
    "crittr": Simulator,   # key string → class
}

def make(name: str, **kwargs):
    """Factory to create environments by name."""
    if name not in _registry:
        raise KeyError(f"Environment '{name}' not found. Available: {list(_registry.keys())}")
    return _registry[name](**kwargs)

__all__ = ["Simulator", "Creature", "make"]
