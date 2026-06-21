# Fluent Automation

This folder contains Fluent setup, meshing, execution, and ASCII extraction helpers.

For objective diagnostics, `extract.py` is the key file: it converts Fluent ASCII exports into `RunResult` coordinates and fields. Any unit convention used by the objective must match what the case DNS loader returns.
