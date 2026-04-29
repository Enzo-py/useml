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
import useml

# 1. Initialize useml, the vault is the storage of useml
useml.init(vault_path=".vault")

# 2. Focus on your project (Automatically creates it if it doesn't exist)
useml.focus("mnist_classifier") # only for creation: useml.new("my-mnist_classifier")

# 3. Track your components
# From now on, useml knows about 'net'. Any commit will include it.
from mymodel.classifier import Classifier
model = Classifier(nb_classes=10, nb_layers=2)
useml.track("net", model)

# 4. Commit your progress
# This archives: source code, weights for 'net', and your custom metadata.
snap = useml.commit(message="Initial baseline", lr=1e-3, accuracy=0.92)
print(f"Committed snapshot: {snap.id}")

# 5. Review project history
# show() gives a nice CLI dashboard of your progress
useml.show()
```

Now from any other script you can have access back to this snapshot (even without the source code):

```python
import useml
useml.init(".vault")
useml_project = useml.focus("my-project")
useml.show()

# 6. Seamless isolation with mount()
# You can mount your project code with the archived version.
useml.mount("\\latest")
useml.show()

# 7. Access to the archived code via useml.workdir
from useml.workdir.mymodel.classifier import Classifier as ArchivedClassifier
model: ArchivedClassifier = useml.load("net") # load the weights of the archived model

# 8. Unmounting the snapshot to end the isolation via useml.workdir
useml.mount("\\current")

# 9. Loading without isolation
model = useml.load("net", _from="\\lastest")
```

## Installation
```bash
pip install useml
```

## Why useml?
Most ML projects fail because of "hidden technical debt" in the pipeline. useml enforces a clean structure from the first line of code, ensuring that every experiment you run is 100% reproducible and deployable.


