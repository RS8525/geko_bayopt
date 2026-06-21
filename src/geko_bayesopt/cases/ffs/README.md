# FFS Case

This folder contains the forward-facing-step case adapter.

`case.py` builds the FFS `CaseConfig`, applies velocity-inlet and pressure-outlet boundary conditions, and loads the LES/DNS CSV fields used by the objective. This case currently returns DNS fields in the CSV's native dimensional units.
