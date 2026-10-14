.PHONY: help setup train predict test lint format clean

help:           ## show this list of commands
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

setup:          ## install everything the project needs
	uv sync --all-groups

train:          ## train the model and save it to models/
	uv run python -m src.train

predict:        ## use the saved model to predict on the test split
	uv run python -m src.predict

test:           ## run the tests
	uv run pytest

lint:           ## point out mistakes and messy code
	uv run ruff check .

format:         ## tidy up the code formatting automatically
	uv run ruff format .
	uv run ruff check --fix .

clean:          ## delete temporary files
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
