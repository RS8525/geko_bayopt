# Test Plan: NM/FD Canonical-Defaults Adaptation in `optimizer.py`

Hand this file to Claude (or work through it manually) on a machine with a
working project venv (numpy, scikit-optimize, pytest). It describes a code
change made on 2026-07-05 that could only be syntax-checked on the machine it
was written on, what behavior it must have, how to verify it, and how to
reverse it if something is wrong.

---

## 1. What was changed and why

`NelderMeadOptimizer` and `FiniteDifferenceOptimizer` in
`src/geko_bayesopt/optimizer.py` used to carry their own hard-coded GEKO
defaults dict `{geko_csep: 1.75, geko_cnw: 0.5, geko_cmix: 0.0, geko_cwall: 0.9}`.
This was outdated: `geko_cwall` is not a supported coefficient, and the
supported `geko_cjet` / `geko_cturb` / `geko_ccorner` were missing, so any
NM/FD-based run tuning those coefficients crashed with a `KeyError`.

Both optimizers now resolve their starting values from the canonical module
`src/geko_bayesopt/geko_defaults.py`
(`csep=1.75, cnw=0.5, cmix=0.0, cjet=1.0, ccorner=1.0, cturb=2.0`),
which was already the single source of truth used elsewhere in the package.

### Exact edits (all in `src/geko_bayesopt/optimizer.py`)

1. **New import** near the top, after the `.config` import:

   ```python
   from .geko_defaults import defaults_for_parameters
   ```

2. **`NelderMeadOptimizer._build_initial_simplex`**: the local
   `all_defaults = {...}` dict was replaced by

   ```python
   defaults_map = defaults_for_parameters(self.parameters)
   defaults = [defaults_map[parameter.name] for parameter in self.parameters]
   ```

3. **`FiniteDifferenceOptimizer`**: the class attribute `_GEKO_DEFAULTS = {...}`
   was deleted; `ask()` now resolves the initial base via

   ```python
   defaults = defaults_for_parameters(self.parameters)
   self._pending_x = np.clip(
       [defaults[p.name] for p in self.parameters],
       self.bounds[:, 0], self.bounds[:, 1],
   )
   ```

4. **`FiniteDifferenceOptimizer.test_config`**: a build-time validation call
   `defaults_for_parameters(self.parameters)` was added at the top (after the
   `n_dim` check). Reason: FD's defaults lookup normally happens in `ask()`,
   but `hybrid_bayes_fd` injects its own base and never takes that branch, so
   without this line an invalid parameter name would slip through the build
   for that hybrid.

### Intended behavior contract

- Runs tuning `geko_csep` / `geko_cnw` / `geko_cmix` are **bit-for-bit
  unchanged** (same defaults values as before).
- `geko_cjet`, `geko_ccorner`, `geko_cturb` are now valid for every optimizer
  kind, including all hybrids.
- `geko_cwall` (and any other unknown name) now fails **at build time** with a
  `ValueError` from `defaults_for_parameters` ("No canonical GEKO default is
  defined for parameter(s): ..."), for all six NM/FD-based kinds. Previously
  this was a `KeyError`, raised at a less predictable moment.
- BO (`skopt_gp`) and PSO never used the defaults dict and must be unaffected.

---

## 2. How to test

Run everything from the project root with the project venv.

### 2.1 Existing test suite

```bash
python -m pytest tests/ -x -q
```

Most relevant files: `test_experiment_baseline.py`, `test_bayesopt_adapter.py`,
`test_config_baseline.py`. Everything that passed before the change must still
pass; there should be no new failures anywhere.

### 2.2 Regression: synthetic benchmark unchanged

The benchmark tunes only `geko_csep` / `geko_cnw`, so its trajectories must be
identical to the committed plots:

```bash
python optimizer_visualization/benchmark_individual.py
git diff --stat optimizer_visualization/plots/individual/
```

Expected: the PNGs are pixel-identical (empty diff), or at minimum every
best-found annotation matches the previous figures exactly.

### 2.3 New-capability spot checks

Run this snippet (e.g. save as a scratch file and execute):

```python
import sys; sys.path.insert(0, "src")
from geko_bayesopt.config import ParameterSpec, OptimizerSection
from geko_bayesopt.optimizer import build_optimizer

P_NEW = [ParameterSpec(name="geko_cjet",  low=0.0, high=2.0),
         ParameterSpec(name="geko_cturb", low=0.5, high=3.0)]
P_BAD = [ParameterSpec(name="geko_cwall", low=0.0, high=2.0)]
KINDS = ["nelder_mead", "finite_differences", "hybrid_nm_bayes",
         "hybrid_fd_bayes", "hybrid_bayes_nm", "hybrid_bayes_fd"]

def sec(kind):
    return OptimizerSection(kind=kind, stopping_criteria={"n_calls": 24},
                            kind_specific_options={})

# (a) new names build and start at the canonical defaults
nm = build_optimizer(sec("nelder_mead"), P_NEW)
assert nm.ask() == [0.9, 2.0], nm.ask()   # cjet 1.0 - 0.1 offset, cturb 2.0
fd = build_optimizer(sec("finite_differences"), P_NEW)
assert fd.ask() == [1.0, 2.0], fd.ask()   # FD base = clipped defaults
for kind in KINDS:
    build_optimizer(sec(kind), P_NEW)     # must not raise
print("new names: OK")

# (b) unknown names fail at build time with ValueError, for every kind
for kind in KINDS:
    try:
        build_optimizer(sec(kind), P_BAD)
    except ValueError as e:
        assert "geko_cwall" in str(e), (kind, e)
    else:
        raise AssertionError(f"{kind} accepted geko_cwall")
print("unknown names rejected at build time: OK")
```

Note on the first assert: the NM startup simplex offsets dimension 0 by
-0.1 first, so the first ask for `(cjet, cturb)` is `[0.9, 2.0]`.

### 2.4 Resume-by-replay sanity (optional but recommended)

If there is an existing test for resume equivalence, run it. Otherwise: run
any cheap NM or FD loop for ~10 tells, rebuild a fresh optimizer, replay the
same (x, y) pairs through `tell()` only, and confirm the next `ask()` matches
the uninterrupted run. The change touches only where the defaults come from,
not the state machines, so any deviation here means something went wrong.

---

## 3. How to reverse the change

### If the change is still uncommitted

```bash
git diff src/geko_bayesopt/optimizer.py     # inspect first
git checkout -- src/geko_bayesopt/optimizer.py
```

### If it has been committed

```bash
git log --oneline -- src/geko_bayesopt/optimizer.py   # find the commit
git revert <commit>
```

### Manual reversal (if git is not an option)

In `src/geko_bayesopt/optimizer.py`:

1. Delete the import line `from .geko_defaults import defaults_for_parameters`.
2. In `NelderMeadOptimizer._build_initial_simplex`, replace

   ```python
   defaults_map = defaults_for_parameters(self.parameters)
   defaults = [defaults_map[parameter.name] for parameter in self.parameters]
   ```

   with

   ```python
   all_defaults = { "geko_csep": 1.75, "geko_cnw": 0.5, "geko_cmix": 0.0, "geko_cwall": 0.9 }

   defaults = [all_defaults[parameter.name] for parameter in self.parameters]
   ```

3. In `FiniteDifferenceOptimizer`, re-add the class attribute directly above
   `ask()`:

   ```python
   _GEKO_DEFAULTS = {"geko_csep": 1.75, "geko_cnw": 0.5, "geko_cmix": 0.0, "geko_cwall": 0.9}
   ```

   and restore `ask()` to

   ```python
   def ask(self) -> list[float]:
       if self._pending_x is None:
           self._pending_x = np.clip(
               [self._GEKO_DEFAULTS[p.name] for p in self.parameters],
               self.bounds[:, 0], self.bounds[:, 1],
           )
       return list(self._pending_x)
   ```

4. In `FiniteDifferenceOptimizer.test_config`, delete the
   `defaults_for_parameters(self.parameters)` line and the comment above it.

Be aware that reversing also restores the old limitation: `geko_cjet`,
`geko_ccorner`, and `geko_cturb` will crash NM/FD-based optimizers again, and
the documentation (`optimizers_readme.md`, `optimizers_documentation.md`,
`optimizer_report.tex`) would then be ahead of the code and need to be
rolled back as well.
