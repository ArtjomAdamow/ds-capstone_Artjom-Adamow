# What are all these tools?

This repo comes with a handful of tools you may not have met before. None of them are
complicated, and none of them are things you need to master. This page explains, for each one:
**what it is**, **why it is in this repo**, and **what you actually have to do**.

Short version if you only read one line: run `uv sync --all-groups` once, then put `uv run` in
front of any Python command. Everything else here is optional comfort.

---

## uv: the only tool you install yourself

**What it is.** A tool that manages Python itself, your project's packages, and the virtual
environment that holds them. It replaces the older combination of pyenv, `python -m venv` and
pip. It is one program instead of three, and it is very fast.

**Why it is here.** Because "it works on my machine" is the classic group-project disaster. uv
writes down the exact version of every package in a file called `uv.lock`, which is committed to
Git. When your teammate runs `uv sync`, they get identical packages. No more "but it worked
yesterday".

**What you do.**

Install it once per computer:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then restart your terminal and check it with `uv --version`.

After that, in the project, two commands cover almost everything:

```bash
uv sync --all-groups      # install everything the project needs
uv run <any command>      # run something using those packages
```

**Why `uv run`?** A command like `python train.py` uses whatever Python your terminal happens to
find, which may be the wrong one with the wrong packages. `uv run python train.py` always uses
this project's environment, and quietly installs anything missing first. It saves you from
activating environments and from a whole category of confusing errors.

More detail, including how to add packages: [UV_SETUP.md](UV_SETUP.md).

---

## The virtual environment (`.venv/`)

**What it is.** A folder inside the project that holds this project's own copy of Python and its
packages, separate from the rest of your computer.

**Why it is here.** So that installing pandas 2.3 for the capstone cannot break a different
project that needs pandas 1.5. Each project gets its own box.

**What you do.** Nothing. `uv sync` creates it, `uv run` uses it, and Git ignores it. You do not
need to activate it, and you should never commit it. If it ever feels broken, delete the whole
`.venv/` folder and run `uv sync --all-groups` again. Nothing of value is lost.

---

## pyproject.toml and uv.lock

**What they are.** `pyproject.toml` is your shopping list: the packages this project needs,
written by hand (or by `uv add`). `uv.lock` is the receipt: the exact versions that were actually
installed, written automatically.

**Why they are here.** The shopping list says "pandas". The receipt says "pandas 2.3.1, which
needed numpy 2.1.0, which needed ...". Keeping both means everyone installs the same thing.

**What you do.** To add a package, do not use pip. Use:

```bash
uv add seaborn            # a package your analysis needs
uv remove seaborn         # changed your mind
```

This updates both files. Commit them together.

In `pyproject.toml` you will see two lists. `dependencies` are packages your project needs to
actually run (pandas, scikit-learn). The `dev` group holds tools you only need while working
(the test runner, the formatter). The split keeps things tidy; you do not have to think about it.

---

## ruff: formatter and linter

**What it is.** Two things in one tool.

A **formatter** rewrites your code into a consistent style: where the line breaks go, single or
double quotes, how much indentation. It does not change what your code does, only how it looks.

A **linter** reads your code and points out likely mistakes: a variable you assigned but never
used, a package you imported but never called, a name that shadows something else.

**Why it is here.** Two reasons. First, when four people write code in one repo, automatic
formatting means nobody argues about style and your Git diffs show real changes instead of
reshuffled whitespace. Second, the linter catches small bugs before you spend twenty minutes
hunting them.

**What you do.** Occasionally, and definitely before you hand in:

```bash
make format     # tidy the code up automatically
make lint       # list the things that look suspicious
```

`make format` is safe: it only reformats, it never changes your logic. `make lint` just prints a
list, it changes nothing.

**Do I have to?** No. Nothing breaks if you never run it. Your code will just be messier to read
for whoever reviews it.

---

## pytest: running tests

**What it is.** A tool that finds functions whose names start with `test_` and runs them. If a
function raises an error, the test fails, and pytest tells you which one and why.

**Why it is here.** Not to make you write a big test suite. The point is narrower: when you write
a function that cleans a column or builds a feature, a three-line test proves it does what you
meant, and keeps proving it after you change something else two weeks later. The alternative is
re-running the whole notebook and eyeballing the output.

**What you do.** Run them:

```bash
make test
```

A test is just a function with an `assert` in it. Look at `tests/test_features.py`:

```python
def test_drop_columns_removes_the_listed_columns():
    result = drop_columns(sample_df(), ["id"])
    assert "id" not in result.columns
```

That is the whole idea: call your function, then state what should be true about the result.

---

## nbdime: for notebook merge conflicts

**What it is.** A tool that shows differences between Jupyter notebooks in a readable way.

**Why it is here.** A `.ipynb` file is not really "a notebook" to Git, it is a large JSON file
that also contains every output and a pile of metadata. So when two teammates edit the same
notebook and you merge, Git shows you a wall of JSON with conflict markers in the middle, which
is close to unreadable. nbdime understands the notebook format and shows you the actual cells
side by side, so you can pick the ones you want.

**What you do.** Nothing, until it happens. If you want to be prepared, turn it on once after
setup:

```bash
uv run nbdime config-git --enable
```

Then if a merge conflict in a notebook ever hits you:

```bash
uv run nbdime mergetool
```

A browser window opens showing both versions. Choose the cells you want, save, then
`git add <notebook>.ipynb` and commit.

**Better still: avoid the conflict.** Agree in your team that one person at a time edits a given
notebook, or give each person their own notebook file. Merge conflicts you never create are the
easiest kind to resolve.

---

## ipykernel: notebooks in VS Code

**What it is.** The piece that lets an editor like VS Code run Python code inside a notebook.

**Why it is here.** So that after `uv sync --all-groups`, this project's environment appears in
VS Code's kernel picker and your notebook cells run with the project's packages.

**What you do.** Nothing, except pick the `.venv` kernel the first time VS Code asks which one to
use. If the notebook says it cannot find pandas, you almost certainly picked the wrong kernel.

---

## joblib: saving your trained model

**What it is.** A small library that writes a Python object to a file and reads it back.

**Why it is here.** Training a model can take minutes or hours. You do not want to retrain it
every time you want a prediction. `train.py` saves the trained model to `models/model.joblib`,
and `predict.py` loads it from there. It is like `pickle`, but better at handling the large
numeric arrays that scikit-learn models are full of.

**What you do.** Nothing. It is two lines inside `train.py` and `predict.py`. Just know that the
file in `models/` is your trained model, and that Git ignores it on purpose (model files get big,
and they can always be regenerated with `make train`).

---

## make and the Makefile

**What it is.** `make` is an old and very widespread tool for running saved commands. The
`Makefile` in this repo is just a list of shortcuts.

**Why it is here.** `uv run python -m src.train` is a lot to type and easy to get wrong.
`make train` is not.

**What you do.** Use it or ignore it, your choice. Every shortcut and the full command it stands
for are listed in the [README](README.md#everyday-commands), so you can always type the long form
instead. `make help` prints the list.

`make` is pre-installed on macOS and Linux. On Windows it usually is not: either install it
(`winget install GnuWin32.Make`) or simply use the full commands, which are identical.

---

## scikit-learn Pipelines, briefly

**What it is.** A `Pipeline` glues preprocessing steps and the model together into one object.
This repo uses one in `src/train.py`:

```python
Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
```

**Why it is here.** When you call `.fit()` on the pipeline, the scaler learns from the training
data and the model is fitted on the scaled result. When you later call `.predict()`, the same
scaler is applied automatically. Saving the pipeline saves the preprocessing along with the
model, so `predict.py` cannot forget a step. Scaling your test data with statistics learned from
the test data is a classic, invisible mistake, and a pipeline makes it hard to make.

---

## Still confused about something?

That is normal, and it is a good question for your coach. These tools exist to remove problems
you have not had yet, which is exactly why they can feel like overhead at first. You can get
through a whole capstone using only these commands:

```bash
uv sync --all-groups                 # set up, once
uv run python -m src.train           # train your model
```

plus the notebook in VS Code. Everything else is there for when you want it.
