"""BUNDLE: a finished experience exported as a standalone folder, to archive the version
that worked. Separate from experiences on purpose: experiences stay light; a bundle is
made once, at the end, and depends on nothing outside itself but Python and the packages
it lists.

    useml.exp.bundle("vision/baseline", "archive/baseline-final")   # or archive=True: .tar.gz
    useml exp bundle vision/baseline -o archive/baseline-final

    archive/baseline-final/
        README.md          what it is, how to load and rerun it
        manifest.json      experience, commit, steps, datasets, versions, sha256 of every file
        load.py            from load import get; model = get("model")
        run.py             python run.py             (checks: every step is already done)
                           python run.py --from-scratch   (recomputes everything)
        useml/             a copy of useml itself: no `pip install useml` needed
        code/              the project code EXACTLY as it ran: commit + uncommitted diff +
                           untracked sources (from the launch snapshot)
        configs/           the config library, when it is not inside the project
        experiences/<name> declaration, outputs of every step, logs, code snapshots
        data/              the processed datasets the steps read — only their rows
        environment/       packages.txt (name==version), python, platform, GPUs

DATA (`data=`): "rows" copies the rows the experience reads (a subset of the table);
"none" copies nothing and the manifest says how to rebuild them; "auto" (default) copies
them up to `max_data` (2 GB), else behaves like "none" and says so.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from .. import _progress


class BundleError(RuntimeError):
    pass


def _walk_recipes(value, kind: str):
    if isinstance(value, dict):
        if set(value) == {kind}:
            yield value[kind]
            return
        for item in value.values():
            yield from _walk_recipes(item, kind)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_recipes(item, kind)


def _latest_snapshot(folder: Path, steps: list[str]) -> tuple[Path, list[str]]:
    snapshots = []
    for name in steps:
        try:
            done = json.loads((folder / "steps" / name / "done.json").read_text())
        except (OSError, ValueError):
            continue
        if done.get("snapshot"):
            snapshots.append(done["snapshot"])
    distinct = sorted(set(snapshots))
    if not distinct:
        candidates = sorted((folder / "code").glob("*/code.json")) if (folder / "code").exists() else []
        if not candidates:
            raise BundleError(f"{folder.name}: no code snapshot (it ran before useml recorded "
                              f"code); run it once more to record one")
        return candidates[-1].parent, []
    return folder / "code" / distinct[-1], distinct


def _export_code(snapshot: Path, destination: Path) -> dict:
    """Rebuild the code of `snapshot` under `destination`."""
    info = json.loads((snapshot / "code.json").read_text())
    root = Path(info["root"])
    destination.mkdir(parents=True, exist_ok=True)
    git = info.get("git")
    if git:
        archive = subprocess.run(["git", "archive", "--format=tar", git["commit"]], cwd=root,
                                 capture_output=True)
        if archive.returncode != 0:
            raise BundleError(f"git archive {git['commit'][:10]} failed in {root}: "
                              f"{archive.stderr.decode(errors='replace').strip()}")
        import io

        with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as bundle:
            if hasattr(tarfile, "data_filter"):
                bundle.extractall(destination, filter="data")
            else:                                         # pragma: no cover
                bundle.extractall(destination)
        patch = snapshot / "diff.patch"
        if patch.exists() and patch.read_text().strip():
            applied = subprocess.run(["git", "apply", "--whitespace=nowarn", str(patch)],
                                     cwd=destination, capture_output=True, text=True)
            if applied.returncode != 0:
                raise BundleError(f"the uncommitted changes of the run do not apply on "
                                  f"{git['commit'][:10]}: {applied.stderr.strip()}")
        untracked = snapshot / "untracked"
        if untracked.exists():
            shutil.copytree(untracked, destination, dirs_exist_ok=True)
        return {"source": "git", "commit": git["commit"], "dirty": git["dirty"],
                "untracked": git["untracked"], "untracked_skipped": git["untracked_skipped"]}
    files = snapshot / "files"
    if files.exists():
        shutil.copytree(files, destination, dirs_exist_ok=True)
    return {"source": "copied folders (no git)", "copied": info.get("copied", [])}


def _copy_useml(destination: Path) -> None:
    import useml

    source = Path(useml.__file__).resolve().parent
    shutil.copytree(source, destination / "useml",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def _export_dataset(name: str, data_root: Path) -> dict:
    """Copy the rows `name` reads into a table under `data_root`."""
    from .. import data
    from ..data import _catalog
    from ..data._table import Table

    view = data.load(name)
    record = _catalog.record(name)
    table = Table(data_root / "tables" / record["table"])
    if not table.exists():
        table.write_manifest(dict(view.table.manifest))
    rows = view.rows
    if (rows < 0).any():
        raise BundleError(f"dataset {name!r}: {int((rows < 0).sum())} rows are missing from "
                          f"its table; process it again before bundling")
    import numpy as np

    unique, first = np.unique(rows, return_index=True)       # each table row once
    present = table.locate(view.digests[first])
    with table.lock():
        writer = table.writer()
        todo = first[present < 0]
        for start in _progress.bar(range(0, len(todo), 65536), desc=f"copying {name}",
                                   unit="chunk"):
            positions = todo[start:start + 65536]
            values = view.get(positions)
            if not isinstance(values, dict):
                values = {next(iter(view.outputs)): values}
            writer.add(view.digests[positions], values)
        writer.flush()
    names = data_root / "names"
    names.mkdir(parents=True, exist_ok=True)
    np.save(names / f"{name}.digests.npy", np.asarray(view.digests))
    (names / f"{name}.json").write_text(json.dumps(record, indent=1, default=str))
    return {"name": name, "rows": len(view), "unique_rows": int(len(unique)),
            "table": record["table"], "identity": record.get("identity")}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle(name: str, destination=None, *, data: str = "auto", max_data="2GB",
           checkpoints: bool = False, archive: bool = False, partial: bool = False) -> Path:
    """Export the experience `name` as a standalone folder (or `.tar.gz` with
    `archive=True`). See the module."""
    from .. import config, __version__
    from ..store import human, parse_size
    from ._experience import open_experience, root as experiences_root

    experience = open_experience(name)
    steps = experience.steps()
    unfinished = [step["name"] for step in steps if not step["done"]]
    if unfinished and not partial:
        raise BundleError(f"{name}: steps not done {unfinished}; finish the run, or pass "
                          f"partial=True to bundle what exists")
    if data not in ("auto", "rows", "none"):
        raise ValueError("data must be 'auto', 'rows' or 'none'")
    destination = Path(destination or Path.cwd() / "bundles" /
                       f"{name.replace('/', '__')}-{time.strftime('%Y%m%d-%H%M%S')}")
    if destination.exists() and any(destination.iterdir()):
        raise BundleError(f"{destination} exists and is not empty")
    destination.mkdir(parents=True, exist_ok=True)
    spec = experience.spec
    manifest = {"experience": name, "created": time.time(), "useml": __version__,
                "steps": steps, "warnings": []}
    try:
        with _progress.step("copying the code as it ran"):
            snapshot, distinct = _latest_snapshot(experience.folder,
                                                  [s["name"] for s in steps if s["done"]])
            manifest["code"] = _export_code(snapshot, destination / "code")
            snapshot_info = json.loads((snapshot / "code.json").read_text())
            manifest["pythonpath"] = snapshot_info["pythonpath"]
            if len(distinct) > 1:
                manifest["warnings"].append(
                    f"steps were produced by {len(distinct)} different launches ({distinct}); "
                    f"the code is the latest one's")
            if manifest["code"].get("untracked_skipped"):
                manifest["warnings"].append(
                    f"untracked files too large to snapshot, NOT in the bundle: "
                    f"{manifest['code']['untracked_skipped']}")
        _copy_useml(destination)

        # configs: inside the project → already in code/; else copied
        library = None
        try:
            library = config.library()
        except Exception:                                    # noqa: BLE001
            pass
        if library is not None and any(_walk_recipes(spec["steps"], "$config")):
            project = Path(snapshot_info["root"])
            try:
                inside = library.root.relative_to(project).as_posix()
                manifest["configs"] = {"path": f"code/{inside}"}
            except ValueError:
                shutil.copytree(library.root, destination / "configs",
                                ignore=shutil.ignore_patterns(".*"))
                manifest["configs"] = {"path": "configs"}
            manifest["configs"].update({
                "kinds": {kind: folder.relative_to(library.root).as_posix()
                          for kind, folder in library.kinds.items()},
                "packages": list(library.namespace.packages),
                "default_kind": library.default_kind,
                "meta_fields": list(library.meta_fields),
                "slot_alias": library.slot_alias})
            if library.on_build is not None or library.kind_of is not None:
                manifest["warnings"].append("the config library has on_build / kind_of hooks; "
                                            "the bundle does not carry them")

        # the experience itself (+ steps of other experiences it reads)
        target = destination / "experiences" / name
        ignore = ["*.stale-*", "__pycache__", ".launch-env.json", ".state.lock", "control"]
        if not checkpoints:
            ignore.append("checkpoints")
        shutil.copytree(experience.folder, target, ignore=shutil.ignore_patterns(*ignore))
        for other in {(o["experience"], o["step"]) for o in _walk_recipes(spec["steps"], "$output")}:
            source = experiences_root() / other[0]
            shutil.copy2(source / "experience.json",
                         _mkdir(destination / "experiences" / other[0]) / "experience.json")
            shutil.copytree(source / "steps" / other[1],
                            destination / "experiences" / other[0] / "steps" / other[1],
                            ignore=shutil.ignore_patterns(*ignore))

        # data
        datasets = sorted({recipe["name"] for recipe in _walk_recipes(spec["steps"], "$data")})
        manifest["datasets"] = []
        if datasets:
            from .. import data as data_module

            sizes = {dataset: data_module.load(dataset).nbytes() for dataset in datasets}
            total = sum(sizes.values())
            copy = data == "rows" or (data == "auto" and total <= parse_size(max_data))
            if data == "auto" and not copy:
                manifest["warnings"].append(
                    f"datasets NOT copied (≈ {human(total)} > {max_data}): rebuild them with "
                    f"the processors listed under `datasets`, or bundle with data='rows'")
            for dataset in datasets:
                if copy:
                    manifest["datasets"].append(_export_dataset(dataset, destination / "data"))
                else:
                    record = data_module.record(dataset)
                    manifest["datasets"].append({"name": dataset, "rows": record["rows"],
                                                 "table": record["table"],
                                                 "identity": record.get("identity"),
                                                 "copied": False})

        # environment
        environment = _mkdir(destination / "environment")
        shutil.copy2(snapshot / "packages.txt", environment / "packages.txt")
        (environment / "platform.json").write_text(json.dumps(
            {key: snapshot_info.get(key) for key in ("python", "platform", "host", "gpus",
                                                      "executable")}, indent=1))

        (destination / "run.py").write_text(_RUN.format(name=name, version=__version__))
        (destination / "load.py").write_text(_LOAD)
        (destination / "README.md").write_text(_readme(manifest))
        files = sorted(path for path in destination.rglob("*") if path.is_file())
        manifest["files"] = {path.relative_to(destination).as_posix(): _sha256(path)
                             for path in _progress.bar(files, desc="checksums", unit="file")}
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    size = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
    for warning in manifest["warnings"]:
        _progress.note(f"⚠ bundle: {warning}")
    if archive:
        with _progress.step(f"compressing {human(size)}"):
            tarball = shutil.make_archive(str(destination), "gztar", destination.parent,
                                          destination.name)
        shutil.rmtree(destination)
        _progress.note(f"bundle written: {tarball}")
        return Path(tarball)
    _progress.note(f"bundle written: {destination} ({human(size)})")
    return destination


def verify(path) -> list[str]:
    """Files whose checksum differs from the manifest, or missing: [] if intact."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    problems = []
    for relative, digest in _progress.bar(manifest["files"].items(), desc="verifying",
                                          unit="file"):
        file = path / relative
        if not file.exists():
            problems.append(f"missing: {relative}")
        elif _sha256(file) != digest:
            problems.append(f"changed: {relative}")
    return problems


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _readme(manifest: dict) -> str:
    code = manifest["code"]
    origin = (f"commit {code['commit'][:12]}" + (" + uncommitted changes" if code.get("dirty") else "")
              if code.get("source") == "git" else code.get("source"))
    steps = "\n".join(f"- `{step['name']}` — `{step['function']}`"
                      + (f" ({step['seconds']:.0f} s)" if step.get("seconds") else "")
                      for step in manifest["steps"])
    datasets = "\n".join(f"- `{d['name']}`: {d['rows']} rows"
                         + ("" if d.get("copied", True) else " (NOT copied)")
                         for d in manifest["datasets"]) or "- none"
    warnings = "\n".join(f"- {w}" for w in manifest["warnings"]) or "- none"
    return f"""# {manifest['experience']} — standalone bundle

Made on {time.strftime('%Y-%m-%d %H:%M', time.localtime(manifest['created']))} with useml
{manifest['useml']}. Code: {origin}.

## Load an output

```python
from load import get
model = get("{manifest['steps'][-1]['name'] if manifest['steps'] else 'step'}")
```

## Run again

```bash
pip install -r environment/packages.txt   # the versions that ran (adapt torch/CUDA)
python run.py                  # checks that every step is done
python run.py --from-scratch   # recomputes every step in experiences/<name>.rerun-<date>
```

## Steps

{steps}

## Datasets

{datasets}

## Warnings

{warnings}

`manifest.json` lists every file with its sha256; `PYTHONPATH=. python -m useml exp
verify .` (from this folder) checks them.
"""


_PRELUDE = '''
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "manifest.json").read_text())
sys.path[:0] = [str(HERE)] + [str((HERE / "code" / p).resolve()) for p in MANIFEST["pythonpath"]]
os.environ.setdefault("USEML_NO_AUTO", "1")
os.environ["USEML_EXP_ROOT"] = str(HERE / "experiences")
os.environ["USEML_DATA_ROOT"] = str(HERE / "data")
os.chdir(HERE / "code")

import useml  # noqa: E402  (the copy shipped in this folder)
from useml import config, exp  # noqa: E402

if MANIFEST.get("configs"):
    settings = dict(MANIFEST["configs"])
    config.use(HERE / settings.pop("path"), **settings)
NAME = MANIFEST["experience"]


def get(step):
    """The output of `step`, rebuilt from this bundle."""
    return exp.open(NAME).get(step)
'''

_LOAD = '"""from load import get; get("<step>") — outputs of this bundle."""' + _PRELUDE

_RUN = '''#!/usr/bin/env python
"""Rerun the bundled experience {name} (useml {version}).

    python run.py                  check that every step is done (nothing recomputed)
    python run.py --from-scratch   recompute every step, in a copy of the experience
"""''' + _PRELUDE + '''

if __name__ == "__main__":
    import shutil
    import time

    experience = exp.open(NAME)
    if "--from-scratch" in sys.argv:
        copy = f"{{NAME}}.rerun-{{time.strftime('%Y%m%d-%H%M%S')}}"
        target = HERE / "experiences" / copy
        target.mkdir(parents=True)
        shutil.copy2(experience.folder / "experience.json", target / "experience.json")
        experience = exp.open(copy)
    experience.run(detach=False)
    print(experience.why())
'''
