# FAQ

**Do I have to use every module?**
No. Each module imports on its own. `import useml` only sets cache paths (if a root is
configured) and the download policy.

**Will `import useml` move my existing caches?**
No. A cache variable already pointing outside the home and `/tmp` is kept. A Hugging Face
cache that already holds models is kept, with a warning. Moving is always explicit:
`useml hub scan --move --apply`.

**Can useml downloads be parallel?**
Yes, explicitly: `[hub] parallel = true` in the machine config, or
`useml.hub.allow_parallel()`.

**Where do experiences live?**
In `[tool.useml.exp] root` of the project (default `experiences/`), else `[exp] root` of
the machine config, else `<home>/experiences`. `$USEML_EXP_ROOT` overrides all of them.

**Should I commit `experiences/` to git?**
Commit `experience.json` if you want the declarations in history. `useml init` adds
ignore rules for what runs write: step outputs, logs, code snapshots, bundles.

**Why must step functions be in a module?**
The run happens in another process, which must import them. A notebook cell cannot be
imported.

**I edited a step function; why did nothing rerun?**
Code changes give a warning, not a rerun: rerunning hours of training for a renamed
variable would be worse. To rerun, change an argument, rename the step, or delete its
folder (`exp.open(name).path(step)`).

**How do I run on a given GPU?**
`exp.experience(name, gpu=1)`, or `gpu=[0, 1]`. It sets `CUDA_VISIBLE_DEVICES` for the
run only.

**Can two machines share an experiences folder?**
Yes. `state.json` records the host; on another machine, a running experience is shown as
running without checking its process.

**How big is a processed table?**
rows × numbers per row × bytes per number. Example: 300 000 images × 512 numbers ×
2 bytes (float16) ≈ 0.3 GB. With `int8`, half of that.

**How do I free disk space?**
See "The disk fills up" in [troubleshooting.md](troubleshooting.md).

**Does a bundle need useml installed?**
No: it contains a copy of useml. It needs Python and the packages listed in
`environment/packages.txt`.

**How do I read the docs offline?**
`useml docs` lists the guides; `useml docs <topic>` prints one.
