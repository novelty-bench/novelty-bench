ruff check --select I --fix $(find src/ -name '*.py')
ruff format src/*.py