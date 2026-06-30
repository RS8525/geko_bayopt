# Configs

Experiment JSON files consumed by `geko-opt run <config>`.

## Path resolution

All relative paths inside a config (`dns_path`, `geometry_path`, `fluent_work_dir`,
`results_dir`) are resolved against the **repository root**, not the config file's
location. The repository root is detected at runtime by walking up from the config
file until a directory containing `pyproject.toml` or `.git` is found.

**Things that break this:**

- Deleting or moving `pyproject.toml` or `.git` — the sentinel lookup fails and
  path resolution falls back to `config.parent.parent`, which is wrong for configs
  nested more than two levels under `configs/`.
- Symlinking or copying a config outside the repository tree — the walk-up will
  find a different (or no) sentinel.
- Writing paths as `../../data/...` relative to the config file — these resolved
  correctly only by accident under the old broken root detection. All paths must
  be written relative to the repository root (e.g. `data/dns/...`).

## Sub-folders

| Folder | Purpose |
|--------|---------|
| `optimizer_comparison_configs/1D/` | One-parameter optimizer comparison, periodic hills Re=2800 |
| `optimizer_comparison_configs/2D/` | Two-parameter optimizer comparison, periodic hills Re=2800 |
| `ffs_final/2_param/` | Final FFS two-parameter production runs |
| `ffs_final/all_param/` | Final FFS five-parameter production runs |
| `ffs_retired/` | Historical FFS configs, provenance only — do not use |
