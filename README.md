# useml

**Stop coding plumbing, start training models.**

`useml` is a minimalist framework for PyTorch designed to eliminate the 90% of software engineering overhead in Machine Learning projects. It handles pipelines, versioning, and environment sealing so you can focus on the architecture.

## Key Features

* **AtomicData**: Type-safe data structures for PyTorch.
* **Auto-Versioning**: Every run is automatically linked to a Git commit and config snapshot.
* **The Vault**: Seal your model, schema, and weights into a single, portable artifact.
* **Zero Friction**: If it doesn't simplify your code, it doesn't belong in `useml`.

## Quick Start

```python
import useml

# Your logic here
# trainer = useml.Trainer(model, schema)
# trainer.train(loader)
```

## Installation
```bash
pip install useml
```

## Why useml?
Most ML projects fail because of "hidden technical debt" in the pipeline. useml enforces a clean structure from the first line of code, ensuring that every experiment you run is 100% reproducible and deployable.

