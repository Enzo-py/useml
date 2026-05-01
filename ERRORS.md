# useml Error Catalog

Every error raised by useml has a unique code of the form `UML-NNN`.
Search this file by code to understand the cause and how to fix it.

## Session (UML-1xx)

### `UML-101` — Not connected to a vault.

**Cause** : useml.init() has not been called, so no vault path is known.

**Fix**   : Call useml.init('path/to/vault') before any vault operation.

### `UML-102` — No project is currently focused.

**Cause** : useml.new() or useml.focus() has not been called.

**Fix**   : Call useml.new('name') to create a project or useml.focus('name') to resume one.

### `UML-103` — You have unsaved changes in RAM.

**Cause** : Components were tracked but not committed before switching projects.

**Fix**   : Call useml.commit('message') to save, or useml.stash() to park changes.

## Vault / Project (UML-2xx)

### `UML-201` — A project with this name already exists in the vault.

**Cause** : useml.new() was called with a name that matches an existing project directory.

**Fix**   : Use useml.focus('name') to resume the existing project, or choose a different name.

### `UML-202` — Cannot compare a Project to an unsupported type.

**Cause** : Project.__eq__ received a value that is neither a string nor a Project.

**Fix**   : Compare projects only to strings (project name) or other Project instances.

## Snapshot (UML-3xx)

### `UML-301` — Snapshot directory is not empty — refusing to overwrite.

**Cause** : A snapshot with the same timestamp already exists on disk.

**Fix**   : This is an internal guard; if it fires, check for clock skew or concurrent writes.

### `UML-302` — Snapshot not found.

**Cause** : The requested snapshot tag or folder name does not exist in the project.

**Fix**   : Run useml.show() to list available snapshots, then pass a valid tag.

### `UML-303` — Invalid snapshot tag.

**Cause** : The tag did not match any known format (\latest, \head~N, or a folder name).

**Fix**   : Use '\\latest', '\\head~N' (N ≥ 0), or the exact snapshot folder name.

### `UML-304` — Weights file not found for the requested component.

**Cause** : The snapshot was saved without weights for this component, or the file was deleted.

**Fix**   : Re-commit the model, or load from a snapshot that contains the component.

### `UML-305` — Failed to load weights — code/weights mismatch.

**Cause** : The model architecture changed after the snapshot was saved.

**Fix**   : Mount the snapshot (useml.mount()) to import the original class, then call useml.load().

### `UML-306` — Snapshot has no source/ directory.

**Cause** : The snapshot was saved before source archiving was introduced, or source saving failed.

**Fix**   : Re-commit the model with the current version of useml to capture sources.

## Import / Workdir (UML-4xx)

### `UML-401` — No snapshot is mounted.

**Cause** : A useml.workdir.* import was attempted before calling useml.mount().

**Fix**   : Call useml.mount('\\latest') (or another tag) before importing from useml.workdir.

### `UML-402` — Module not found in the mounted snapshot.

**Cause** : The requested module path does not exist in the snapshot's source/ directory.

**Fix**   : Check the module name with useml.debug_imports(), then correct the import path.

### `UML-403` — Cannot import class from the current working directory.

**Cause** : The module path stored in the snapshot manifest is no longer importable.

**Fix**   : Ensure the module is on sys.path, or mount the snapshot with useml.mount().

## Training (UML-5xx)

### `UML-501` — Unknown loss function key.

**Cause** : The string passed to Config(loss=...) is not in the built-in loss registry.

**Fix**   : Use one of the built-in keys ('cross_entropy', 'mse', 'bce', 'l1'), or pass an nn.Module.

### `UML-502` — Invalid loss type.

**Cause** : Config.loss received a value that is not a string, nn.Module subclass, instance, or callable.

**Fix**   : Pass a string key, an nn.Module subclass, an nn.Module instance, or a callable.

### `UML-503` — Unknown optimizer key.

**Cause** : The string passed to Config(optimizer=...) is not in the built-in optimizer registry.

**Fix**   : Use one of the built-in keys: 'adam', 'adamw', 'sgd'.

## Dataset (UML-6xx)

### `UML-601` — torchvision is not installed.

**Cause** : A built-in dataset was requested but torchvision is missing from the environment.

**Fix**   : Run: pip install torchvision

### `UML-602` — The 'datasets' package is not installed.

**Cause** : An 'hf:' dataset was requested but the HuggingFace datasets library is missing.

**Fix**   : Run: pip install datasets

### `UML-603` — Unknown dataset name.

**Cause** : The string passed to train(dataset=...) is not a built-in name and does not start with 'hf:'.

**Fix**   : Use a built-in name ('mnist', 'cifar10', …), 'hf:<name>', or a torch Dataset instance.

### `UML-604` — Invalid dataset type.

**Cause** : dataset must be a string or a torch.utils.data.Dataset, but received something else.

**Fix**   : Pass a string name or a torch Dataset instance to train() / load_dataset().

