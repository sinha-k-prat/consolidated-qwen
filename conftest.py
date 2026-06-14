# Presence of this file at the repo root puts the root on sys.path during pytest
# collection, so `from model import ...` in tests/ resolves regardless of how
# pytest is invoked (bare `pytest` vs `python -m pytest`).
