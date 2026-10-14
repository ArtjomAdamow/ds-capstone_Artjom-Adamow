# Data Science Capstone Template

A ready-made folder structure for your capstone project, so you can start with the data science
instead of with the setup.

Everything here already works: you can install it, train an example model, and see a result in
about five minutes. Then you swap the example out for your own project, step by step.

New to the tools used here (uv, ruff, pytest, nbdime)? Read
**[TOOLS.md](TOOLS.md)** first. It explains each one in plain language: what it is, why it is in
this repo, and what you actually have to type.

## Getting started

You need one thing installed: **uv**. If you do not have it yet, see
[TOOLS.md](TOOLS.md#uv-the-only-tool-you-install-yourself).

**1. Make your own copy.** On GitHub, click the green **Use this template** button at the top of
this page. That creates a fresh repository for your team.

**2. Download it to your computer:**

```bash
git clone git@github.com:<your-org>/<your-repo>.git
cd <your-repo>
```

**3. Install everything:**

```bash
uv sync --all-groups
```

This one command installs Python (yes, Python itself, if you need it), creates the project's
virtual environment in a `.venv/` folder, and installs pandas, scikit-learn and the rest. It
takes a minute the first time.

**4. Check it works:**

```bash
uv run python -m src.train
```

You should see a few log lines ending with a trained model and two scores. That is the example
pipeline running end to end. Now you know your setup is fine.

**5. Open the notebook.** Open the project folder in VS Code, open
`notebooks/EDA-and-modeling.ipynb`, and when VS Code asks which kernel to use, pick the one from
`.venv`. That is the environment you just installed.

## What is in this repo

| Folder or file | What it is for |
|---|---|
| `notebooks/` | Your exploration and analysis. `EDA-and-modeling.ipynb` is an empty notebook with the sections already laid out. |
| `src/` | Python code you want to reuse. See the table below. |
| `tests/` | Checks that your code in `src/` does what you think. One example is included. |
| `data/` | Your data files. Git ignores what you put here, so nothing large or private ends up on GitHub. |
| `models/` | Trained models get saved here. Also ignored by Git. |
| `images/` | Plots you want to keep, for example for your final presentation. |
| `pyproject.toml` | The list of packages the project uses. |
| `Makefile` | Shortcuts for the long commands. `make train` instead of `uv run python -m src.train`. |
| `TOOLS.md` | Plain-language explanation of every tool in here. |
| `UV_SETUP.md` | The longer reference for uv, for when you need to add a package or something breaks. |

Inside `src/`:

| File | What it does |
|---|---|
| `config.py` | Settings in one place: folder paths, the name of the column you predict, the random seed. Change settings here, not scattered through the code. |
| `data.py` | Loads your data and splits it into a training set and a test set. |
| `features.py` | Cleaning and feature engineering. |
| `train.py` | Trains the model, prints how good it is, saves it to `models/`. |
| `predict.py` | Loads a saved model and makes predictions on new data. |

## Everyday commands

You can use either column. They do exactly the same thing.

| Short version | Full version | What it does |
|---|---|---|
| `make setup` | `uv sync --all-groups` | Install or update the packages |
| `make train` | `uv run python -m src.train` | Train the model and save it |
| `make predict` | `uv run python -m src.predict` | Use the saved model to predict |
| `make test` | `uv run pytest` | Run the tests |
| `make lint` | `uv run ruff check .` | Point out mistakes and messy code |
| `make format` | `uv run ruff format .` | Tidy up the code formatting automatically |
| `make clean` | | Delete temporary files |

`make help` lists them too. `make` is not installed on Windows by default, so if the command is
not found, just use the full version. It is exactly the same command.

> Why `uv run` in front of everything? Because it guarantees the command uses this project's
> packages and not some other Python on your computer. You never have to "activate" anything.

## Turning this into your project

The repo ships with a small example (predicting a diabetes progression score from a built-in
dataset) purely so that everything runs before you have your own data. Replace it in this order,
and keep running `make train` after each step to see that nothing broke:

**1. Your data.** Open `src/data.py` and change `load_data` so it returns your dataset as a
DataFrame. Usually one line:

```python
return pd.read_csv(DATA_DIR / "my_dataset.csv")
```

Then open `src/config.py` and set `TARGET` to the name of the column you want to predict.

**2. Your cleaning and features.** Put your cleaning steps into `engineer_features` in
`src/features.py`. There are two small helpers there to get you going, and you can add your own.

Why there and not in the notebook? Because `train.py` and `predict.py` both call this one
function. If you clean your data in the notebook only, your model will be trained on clean data
and later asked to predict on messy data, and the results will be quietly wrong. This is one of
the most common ways capstone projects go sideways.

**3. Your model.** In `src/train.py`, change `build_model` to the model you want (a random
forest, a classifier, whatever suits your question), and change `evaluate` to metrics that fit.
The example uses RMSE and R2, which only make sense for predicting a number. If you are
predicting a category, you want accuracy, precision, recall or F1 instead.

**4. Your tests.** Add a test to `tests/` for each function you write in `src/features.py`. Copy
the style of the examples that are already there.

**5. This README.** Delete this section and describe your own project: the question, the data,
what you found.

## Working as a team

- Work on a branch, open a pull request, have a teammate look at it before merging.
- Use notebooks to explore. As soon as a piece of code is worth running twice, move it into
  `src/` and import it back into the notebook.
- If you add a package with `uv add <name>`, commit both `pyproject.toml` and `uv.lock` so
  everyone gets the same versions.
- Notebooks cause ugly merge conflicts. [TOOLS.md](TOOLS.md#nbdime-for-notebook-merge-conflicts)
  explains the tool that fixes that.

## Where to look when something breaks

- Setup and package problems: [UV_SETUP.md](UV_SETUP.md#7-troubleshooting)
- "What even is this tool": [TOOLS.md](TOOLS.md)
- [scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html) for the modeling
- [pandas documentation](https://pandas.pydata.org/docs/) for the data wrangling
