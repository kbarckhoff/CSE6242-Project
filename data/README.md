# Data layout

- data/raw/       → local-only raw inputs (NOT committed)
- data/processed/ → local-only intermediate outputs (NOT committed)
- data/samples/   → tiny sample files that are committed for review/tests

> We are storing large files on Google Cloud. Only commit tiny samples needed for code review.
