# Configs

Experiment JSON files select the flow case, objective, optimizer, mesh settings, and Fluent session strategy.

For the current FFS diagnosis, `ffs_csep_v3.json` is the source config. It uses the `ffs` case, optimizes `geko_csep`, and scores `["cp", "Ux", "Uy"]` with the `gedcp` objective.
