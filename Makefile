.PHONY: install test lint train dash

install:
	pip install -r requirements-dev.txt

test:
	pytest

lint:
	ruff check src tests

train:
	python -m src.train

dash:
	streamlit run dashboard/app.py
