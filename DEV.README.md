## Development & Testing

If you want to contribute or test the framework, follow these steps.

### 1. Install for development
Clone the repo and install it in editable mode with dev dependencies:
```bash
pip install -e ".[dev]"
```

### 2. Running Tests
We use pytest for unit testing. The tests are located in the tests/ directory.
To run the full suite:
```bash
pytest -vv
```

### 3. Build & Publish
**Update Version**: Increment version in `pyproject.toml`.

Build the package:
```bash
rm -rf dist/ build/ *.egg-info
python3 -m build
twine check dist/*
```

Upload the new package version:
```bash
twine upload dist/*
```