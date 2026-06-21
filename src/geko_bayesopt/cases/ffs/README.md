# FFS Case

This folder contains the forward-facing-step case adapter.

`case.py` builds the FFS `CaseConfig`, applies velocity-inlet and static-pressure-outlet boundary conditions, and loads the LES/DNS CSV fields used by the objective. This case currently returns DNS fields in the CSV's native dimensional units.

FFS outlet pressure is configured through PyFluent's structured `pressure_outlet[zone].momentum` settings. Do not drive it through the pressure-outlet TUI prompt stream; the prompt sequence is release-sensitive and has produced invalid pressure/profile settings.
