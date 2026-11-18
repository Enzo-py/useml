"""Processors: a function (or class) turning a BATCH of items into one row per item.

    @useml.data.processor("clip-image", version=1, batch_size=64, dtype="float16")
    class ClipImage:
        def __init__(self, model="openai/clip-vit-base-patch32", device="cuda"):
            ...                                   # loads the model: runs ONLY if a row is missing
        def load(self, path):                     # optional, per item, in worker threads
            return Image.open(path).convert("RGB")
        def __call__(self, images):               # a batch → (B, ...) array, or a dict of them
            return embeddings

    @useml.data.processor("caption-length", dtype="int32")
    def caption_length(texts, *, unit="char"):
        return np.array([len(text) for text in texts])

IDENTITY. A table is named by the processor's name, its `version`, its parameters (the
class's constructor arguments or the function's keyword arguments, minus `ignore`, by
default `device` and `batch_size`) and its output dtypes. Bump `version` when the code
changes what it computes: the code is also hashed, and a table built by different code
under the same version raises a warning, never silently mixes the two.

DTYPE. `"float32"`, `"float16"` (default for floats) or `"int8"`: one byte per number plus
one float32 scale per row (symmetric, per row), read back as float32. Integer and boolean
outputs keep their own dtype. A dict gives one dtype per output.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
from dataclasses import dataclass, field

from ._keys import fingerprint

FLOAT_DTYPES = ("float32", "float16", "int8")


@dataclass
class Processor:
    target: object
    name: str
    version: int | str = 1
    batch_size: int = 32
    dtype: object = "float16"
    ignore: tuple = ("device", "batch_size")
    workers: int = 4
    load: object = None
    defaults: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        dtypes = self.dtype.values() if isinstance(self.dtype, dict) else [self.dtype]
        for value in dtypes:
            if value not in FLOAT_DTYPES:
                raise ValueError(f"processor {self.name!r}: dtype {value!r}; use one of "
                                 f"{FLOAT_DTYPES} (integer outputs keep their own dtype)")
        signature = inspect.signature(self.target.__init__ if self.is_class else self.target)
        self.defaults = {name: parameter.default for name, parameter in
                         signature.parameters.items()
                         if parameter.default is not inspect.Parameter.empty
                         and name != "self"}

    @property
    def is_class(self) -> bool:
        return inspect.isclass(self.target)

    def dtype_of(self, output: str) -> str:
        return self.dtype.get(output, "float16") if isinstance(self.dtype, dict) \
            else self.dtype

    def identity(self, params: dict) -> dict:
        """What the rows depend on: name, version, parameters (defaults included, so
        passing a default explicitly does not change the table), dtypes."""
        merged = {**self.defaults, **params}
        kept = {name: value for name, value in merged.items() if name not in self.ignore}
        return {"processor": self.name, "version": self.version, "params": kept,
                "dtype": self.dtype}

    def fingerprint(self, params: dict) -> str:
        return fingerprint(self.identity(params))

    def code_hash(self) -> str:
        try:
            text = inspect.getsource(self.target)
        except (OSError, TypeError):
            return ""
        return hashlib.blake2b(text.encode(), digest_size=8).hexdigest()

    @property
    def import_path(self) -> str | None:
        """`module:qualname`, if another process can import it."""
        module = getattr(self.target, "__module__", None)
        qualname = getattr(self.target, "__qualname__", "")
        if not module or module == "__main__" or "<locals>" in qualname:
            return None
        return f"{module}:{qualname}"

    def instantiate(self, params: dict):
        """The callable that processes a batch. For a class, this runs its constructor
        (loads the model): call it only when something is missing."""
        if self.is_class:
            return self.target(**params)
        target = self.target
        return lambda batch: target(batch, **params)

    def loader(self, instance):
        """The per-item loader: the class's `load` method, or `load=` of the decorator."""
        if self.load is not None:
            return self.load
        return getattr(instance, "load", None) if self.is_class else None

    def __call__(self, *args, **kwargs):
        """Calling the decorated object keeps its normal behaviour."""
        return self.target(*args, **kwargs)

    def __repr__(self) -> str:
        return f"Processor({self.name!r}, version={self.version!r})"


def processor(name: str | None = None, *, version=1, batch_size: int = 32,
              dtype="float16", ignore=("device", "batch_size"), workers: int = 4,
              load=None):
    """Decorate a function or a class as a processor (see the module)."""
    def wrap(target):
        processed = Processor(target, name or target.__name__, version, batch_size, dtype,
                              tuple(ignore), workers, load)
        _REGISTRY[processed.name] = processed
        return processed
    if callable(name):                              # @processor without parentheses
        target, name = name, None
        return wrap(target)
    return wrap


_REGISTRY: dict[str, Processor] = {}


def resolve(value) -> Processor:
    """A Processor, a registered name, or "module:qualname"."""
    if isinstance(value, Processor):
        return value
    if isinstance(value, str) and value in _REGISTRY:
        return _REGISTRY[value]
    if isinstance(value, str) and ":" in value:
        module_name, _, attribute = value.partition(":")
        target = importlib.import_module(module_name)
        for part in attribute.split("."):
            target = getattr(target, part)
        if isinstance(target, Processor):
            return target
        raise TypeError(f"{value} is not decorated with @useml.data.processor")
    raise KeyError(f"unknown processor {value!r}; known: {sorted(_REGISTRY)}")
