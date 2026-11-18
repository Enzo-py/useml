"""useml.exp — experiences declared as a script, run detached, read back in one line.

    from useml import exp

    with exp.experience("vision/baseline") as e:
        trained = e.step(train, data.load("photos-clip"), epochs=10, name="model")
        e.step(report, trained, config.load("resnet_ft"), name="report")

    model = exp.get("vision/baseline/model")      # later, anywhere
    exp.open("vision/baseline").results()          # the numbers of every step
    exp.ls(); exp.running()

A step is an importable function; its arguments are plain values, outputs of earlier
steps, objects from `useml.config`, processed datasets from `useml.data` (by name), or
outputs of other experiences. What it returns is saved by type and loaded back by `get`
(a model built from a config: its config + weights). See `_experience`, `_refs`,
`_runtime`, `_outputs`.
"""
from ._experience import (Builder, Experience, experience, get, ls, open_experience as open,
                          root, running)
from ._bundle import BundleError, bundle, verify
from ._outputs import register_saver
from ._refs import RefError, StepRef
from ._runtime import Checkpointer, Context

__all__ = ["experience", "get", "open", "ls", "running", "root", "Experience", "Builder",
           "StepRef", "RefError", "Context", "Checkpointer", "register_saver",
           "bundle", "verify", "BundleError"]
