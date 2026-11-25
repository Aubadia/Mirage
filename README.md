# Mirage

Mirage is a research project for testing deep learning framework correctness across tensor representations.

## Install from source

Install uv using the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/). Clone or download this repository, open a terminal in its root, and run the following commands.

```sh
uv python install 3.14
uv sync --locked --no-dev --no-editable
uv run --no-sync mirage
```

uv creates the project environment from the lockfile. 

For the development environment, CI checks, and test tools, see [CONTRIBUTING.md](CONTRIBUTING.md).
