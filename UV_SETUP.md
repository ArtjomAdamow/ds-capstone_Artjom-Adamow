# Working with this repo using `uv`

This project is managed entirely by [uv](https://docs.astral.sh/uv/): it installs the Python
interpreter, creates the virtual environment, resolves dependencies and locks them, all in one
tool. You do not need pyenv, conda, or pip.

The [README](README.md) covers the short version, and [TOOLS.md](TOOLS.md) explains in plain
language what uv is and why the project uses it. This file is the detailed reference for when
you need more: adding packages, pinning Python, CI, troubleshooting.

---

## 1. Install uv (once per machine)

**macOS / Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Alternatives: `brew install uv`, `winget install --id=astral-sh.uv`, or `pipx install uv`.
Restart your shell, then check with `uv --version`. Upgrade later with `uv self update`.

---

## 2. Create the environment

```bash
uv sync --all-groups
```

That one command:

1. Reads the required Python version from `.python-version` and `pyproject.toml`.
2. Downloads a matching interpreter if you do not have one.
3. Creates `.venv/` inside the project.
4. Installs the runtime dependencies **and** the `dev` group.
5. Uses `uv.lock`, so everyone on the team gets identical versions.

Variants:

```bash
uv sync                  # runtime dependencies only
uv sync --all-groups     # runtime plus dev tools (what you want while working)
uv sync --frozen         # install exactly what uv.lock says, never re-resolve (use in CI)
```

---

## 3. Running things

`uv run` syncs the environment before running, so you never work against a stale venv and never
have to activate anything:

```bash
uv run python -m src.train     # train the model
uv run python -m src.predict   # score the saved model
uv run pytest                  # tests
uv run ruff check .            # lint
uv run ruff format .           # format
```

The `Makefile` wraps the common ones: `make setup`, `make train`, `make predict`, `make test`,
`make lint`, `make format`, `make clean`.

For notebooks, open them in VS Code and pick `.venv` as the kernel. `ipykernel` is already in the
dev group, so the environment shows up in the kernel picker after `uv sync --all-groups`.

---

## 4. Managing packages

Never call `pip install` again. uv updates `pyproject.toml` and `uv.lock` for you:

```bash
uv add plotly                    # add a runtime dependency
uv add --group dev mypy          # add a dev-only dependency
uv add "pandas>=2.2"             # add with a version constraint
uv remove plotly                 # remove a dependency
```

Upgrading:

```bash
uv lock --upgrade                 # re-resolve everything to the newest compatible versions
uv lock --upgrade-package pandas  # bump a single package
uv sync --all-groups              # apply the new lock to your .venv
```

Inspecting:

```bash
uv pip list                      # installed packages
uv tree                          # dependency tree
```

Commit `pyproject.toml` and `uv.lock` together whenever you change dependencies, so your
teammates get the same environment from a plain `uv sync`.

---

## 5. Python version

`.python-version` pins the interpreter for this project (currently `3.13`) and `pyproject.toml`
declares the supported range (`requires-python = ">=3.11"`). uv honours the pin and downloads
the interpreter if it is missing.

```bash
uv python pin 3.12          # change the pin
uv python list              # see available and installable interpreters
uv python install 3.13      # install an interpreter explicitly
```

---

## 6. What to commit

| File | Commit? | Why |
|---|---|---|
| `pyproject.toml` | Yes | declares the dependencies |
| `uv.lock` | Yes | guarantees reproducible installs for everyone |
| `.python-version` | Yes | keeps the whole team on the same interpreter |
| `.venv/` | No | git-ignored, rebuilt by `uv sync` |
| `data/`, `models/` contents | No | git-ignored, often large or private |

If an external system still needs a `requirements.txt`, generate it rather than hand-writing it:

```bash
uv export --no-dev --format requirements-txt -o requirements.txt
```

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `uv: command not found` | Restart the shell; check `~/.local/bin` (or `%USERPROFILE%\.local\bin`) is on `PATH`. |
| Environment feels stale or has wrong versions | `uv sync --all-groups` reconciles `.venv` with the lock, including removals. |
| Want a clean rebuild | Delete `.venv/` and run `uv sync --all-groups`. |
| Resolution conflict after `uv add` | Loosen the constraint, or run `uv lock --upgrade` to re-resolve the whole graph. |
| `ModuleNotFoundError: No module named 'src'` in a notebook | Run the setup cell at the top; it adds the project root to `sys.path`. |
| VS Code does not list the environment | Reload the window, then pick `.venv/bin/python` (`.venv\Scripts\python.exe` on Windows) in the kernel picker. |
| CI must not change the lock | Use `uv sync --frozen`, which fails if `uv.lock` is out of date. |
