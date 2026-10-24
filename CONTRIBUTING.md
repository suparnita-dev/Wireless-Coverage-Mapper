# Contributing

1. Create a focused branch for the change.
2. Install dependencies with `python -m pip install -r requirements.txt`.
3. Run `python -m pytest -q`.
4. Run Ruff:

   ```text
   python -m ruff check signal_mapper.py test_signal_mapper.py
   python -m ruff format --check signal_mapper.py test_signal_mapper.py
   ```

5. Update tests and documentation when behavior changes.
6. Keep recorded sample data and computed outputs clearly separated from code.

Pull requests should explain the engineering motivation, validation performed,
and whether an output is a measurement or a grid-derived estimate.
