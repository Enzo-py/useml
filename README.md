# useml

**Stop coding plumbing, start training models.**

`useml` is a minimalist framework for PyTorch designed to eliminate the 90% of software engineering overhead in Machine Learning projects. It handles pipelines, versioning, and environment sealing so you can focus on the architecture.

## Key Features

* **AtomicData**: Type-safe data structures for PyTorch.
* **Auto-Versioning**: Every run is automatically linked to a Git commit and config snapshot.
* **The Vault**: Seal your model, schema, and weights into a single, portable artifact.
* **Zero Friction**: If it doesn't simplify your code, it doesn't belong in `useml`.

### Basic Usage

```python
import torch
from useml import Vault

# 1. Initialize your vault (the root storage)
vault = Vault(path="my_vault")

# 2. Get or create a specific Project
project = vault.get_project("mnist_classifier")

# 3. Define your model
model = torch.nn.Linear(10, 2)

# 4. Commit your progress
# This saves: weights.pth + manifest.yaml + metadata.yaml
project.commit(model, message="Initial baseline", lr=1e-3, accuracy=0.92)

# 5. List project history (newest first)
for snap in project.log():
    print(f"[{snap['timestamp']}] {snap['message']} | Acc: {snap['accuracy']}")

# 6. Restore weights from the latest snapshot
latest = project[0]
latest.load_weights(model)
```

## Installation
```bash
pip install useml
```

## Why useml?
Most ML projects fail because of "hidden technical debt" in the pipeline. useml enforces a clean structure from the first line of code, ensuring that every experiment you run is 100% reproducible and deployable.


## Development & Testing

If you want to contribute or test the framework, follow these steps.

### 1. Install for development
Clone the repo and install it in editable mode with dev dependencies:
```bash
pip install -e ".[dev]"
```

### 2. Running Tests
We use pytest for unit testing. The tests are located in the tests/ directory.
To run the full suite:
```bash
pytest -vv
```
