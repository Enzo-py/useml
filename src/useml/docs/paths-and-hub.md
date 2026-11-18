# Caches and downloads: `useml.paths` and `useml.hub`

## One cache root per machine

```toml
# ~/.config/useml/config.toml
root = "/data/me/useml"

[env]                       # optional: a cache that must live elsewhere
HF_HOME = "/data/shared/hf"
```

`$USEML_ROOT` overrides `root`. Without a root (a laptop), library caches stay where they
are and useml keeps its own files in `~/.cache/useml`.

`import useml` then sets these variables, unless one is already set to a folder outside
the home and `/tmp` (a cache already placed on a big disk is never moved):

| Variable | Folder under the root |
|---|---|
| `HF_HOME` | `hub/huggingface` |
| `TORCH_HOME` | `hub/torch` |
| `NLTK_DATA` | `hub/nltk_data` |
| `XDG_CACHE_HOME`, `PIP_CACHE_DIR`, `UV_CACHE_DIR`, `MPLCONFIGDIR`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `CUDA_CACHE_PATH`, `VLLM_CACHE_ROOT` | `cache/<name>` |
| `TMPDIR` | `tmp` |

Three details are handled for you:

- Variables for one part of the Hugging Face cache (`HF_HUB_CACHE`, `TRANSFORMERS_CACHE`…)
  follow `HF_HOME` if they point to an empty folder in the home; if that folder already
  holds models, it is kept and a warning says so.
- If `huggingface_hub` was imported before useml, its paths are realigned.
- The login token of `hf auth login` stays readable after `HF_HOME` moves.

```bash
useml paths        # every variable before → after, and where models already are
```

## One gate for downloads

```python
from useml import hub

path = hub.get("hf://openai/clip-vit-base-patch32")                       # whole repo
path = hub.get("hf://openai/clip-vit-base-patch32", allow=["*.json", "*.safetensors"])
path = hub.get("hf://openai/clip-vit-base-patch32@main/config.json")      # one file
path = hub.get("hf://datasets/ylecun/mnist")                              # dataset repo
path = hub.get("https://example.org/annotations.zip", extract_archive=True)
path = hub.get("gdrive://<file id>")
path = hub.get("gdrive-folder://<folder id>")
```

- **Local first**: a URI already fetched is answered from disk, without network.
  `refresh=True` asks again.
- **Where**: Hugging Face files stay in the Hugging Face cache, so `transformers` and
  `diffusers` find them unchanged. Other files go to `<root>/hub/files/`, or to `dest=`.
- **Interrupted HTTP downloads resume** (a `.part` file and a Range request).
- **Archives** are unpacked once; a member that would escape the folder is refused.
- **A manual download** (rate-limited Google Drive…) is recorded with
  `hub.adopt("gdrive://<id>", "/path/where/I/put/it")`.
- **An expired Hugging Face token** is reported as such, not as "not a valid model
  identifier".

```bash
useml hub ls          # what was downloaded, size, last use
useml hub get <uri>   # download from the shell, print the path
```

## The download policy: one thread at a time

By default only one thread may download at a time, in every library. Four layers enforce
it:

| Layer | What it does |
|---|---|
| 0 | turns off Hugging Face `xet` (it opens parallel connections in Rust, invisible to Python) |
| 1 | makes the thread pools of `huggingface_hub` and `datasets` sequential |
| 2 | refuses an outbound TCP connection from a second thread (`ParallelDownloadError`), and logs it |
| 3 | `useml hub watch -- <command>`: an external process watching the whole process tree |

```bash
useml hub status                          # which layers hold right now
useml hub watch -- python train.py        # run a command under the external monitor
useml hub watch --pid 12345 --no-kill     # watch a running process, alert only
```

To allow parallel downloads: `[hub] parallel = true` in the machine config, or
`useml.hub.allow_parallel()` in code. Loopback connections (`localhost`) are never
counted.
