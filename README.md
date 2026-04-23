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
import useml

# 1. Initialize the session (Sets up the global singleton)
useml.init(vault_path="my_vault")

# 2. Focus on your project (Automatically creates it if it doesn't exist)
useml.focus("mnist_classifier")

# 3. Track your components
# From now on, useml knows about 'net'. Any commit will include it.
model = torch.nn.Linear(10, 2)
useml.track("net", model)

# 4. Commit your progress
# This archives: source code, weights for 'net', and your custom metadata.
snap = useml.commit(message="Initial baseline", lr=1e-3, accuracy=0.92)
print(f"Committed snapshot: {snap.id}")

# 5. Review project history
# show() gives a nice CLI dashboard of your progress
useml.show()

# 6. Seamless isolation with mount()
# Inside this block, your project code is swapped with the archived version.
# No manual re-imports needed, no dirty state leaks.
with useml.mount(snap.id):
    # If your model definition changed in the 'present', 
    # it is reverted to the 'past' version here.
    import my_model_script 
    print(f"Archived code version: {my_model_script.VERSION}")

# 7. Restore weights (Manual or via Snapshot object)
# You can grab a specific snapshot from history
old_snap = useml.log()[0]
old_snap.resume() # Restores weights for all tracked components
```

## Installation
```bash
pip install useml
```

## Why useml?
Most ML projects fail because of "hidden technical debt" in the pipeline. useml enforces a clean structure from the first line of code, ensuring that every experiment you run is 100% reproducible and deployable.


