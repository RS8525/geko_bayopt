# Session State: Optimizer Documentation (hand-off)

State of the optimizer-documentation work as of 2026-07-05, written so a new
Claude session on another machine can continue without re-deriving context.
Read this together with `TESTING_optimizer_defaults.md` (test plan for the one
code change).

---

## 1. Goal of the overall task

Produce a single LaTeX document that serves as the final documentation of the
optimizers in this code base. Work so far happened in two sessions:

1. Draft `optimizer_visualization/optimizer_report.tex` covering (a) a
   configuration reference for every optimizer in
   `src/geko_bayesopt/optimizer.py` and (b) the synthetic benchmark in
   `optimizer_visualization/_benchmark_core.py` / `benchmark_individual.py`
   with all eight plots from `optimizer_visualization/plots/individual/`.
2. Apply the user's section-by-section corrections (done, see below).

**Next major step (explicitly deferred, not started):** merge
`optimizer_visualization/optimizer_comparison_report.tex` (the real-CFD
Periodic Hills Re=2800 comparison) into `optimizer_report.tex`. Instruction
from the user: adapt tonality, style, and formulation of the merged content to
`optimizer_report.tex`, do NOT carry over the style of the comparison report.
Note the comparison report contains two placeholder `\bibitem`s
(`bo_hyperparams_ref`, `pso_future_work_ref`) and a pending 5D-study section;
ask the user how to handle these during the merge.

## 2. Style rules the user has set (binding for all future edits)

- No em-dashes anywhere ("AI-affiliated m-dashes"); restructure sentences or
  use commas/colons instead. En-dashes in names (Nelder--Mead) and numeric
  ranges are fine.
- Do not try to compile the LaTeX to PDF.
- Do not fix or change the plots; content only.
- Every content fix applied to the report must be mirrored into the .md files
  (`src/geko_bayesopt/optimizers_readme.md`,
  `src/geko_bayesopt/optimizers_documentation.md`,
  `optimizer_visualization/benchmark_readme.md`) so future sessions do not
  regenerate outdated statements.
- No mentions of historical implementation artifacts ("an earlier version
  did X", "this was a real bug") in the report; state present-tense rationale
  only.
- If a change needs further specification, ask the user before changing code.

## 3. Work completed

### 3.1 `optimizer_visualization/optimizer_report.tex` (created, then revised)

Structure: Sec. 1 shared architecture (ask/tell, config shape, stopping
criteria, boundary policy, determinism); Sec. 2 reference for all eight kinds
(baseline JSON config, options table, possible issues each); Sec. 3 synthetic
benchmark routine (test functions, harness, initialization table); Sec. 4 all
eight figures with short discussions plus a summary table.

All corrections from the user's review are applied, including:
- Boundary policy: no old-clipping history; states the walk can wander far
  outside the bounds, hard limit is only what Fluent accepts (never happened
  in practice yet).
- Determinism section simplified (no "cooked by skopt" internals).
- NM/FD defaults: csep=1.75, cnw=0.5, cmix=0.0, cjet=1.0, ccorner=1.0,
  cturb=2.0 (cwall removed); parameter-name possible-issues dropped.
- PSO: divisibility only under possible issues; no max_iter-key remark.
- Benchmark section updated to the new hyperparameters (see 3.3).
- Sections 4.5/4.6 (NM->BO, FD->BO) note that the warm-started BO phase
  behaves strongly exploitative rather than explorative, presumably a
  phenomenon of the underlying Gaussian process, to be studied further.

### 3.2 Code change in `src/geko_bayesopt/optimizer.py` (uncommitted)

NM and FD now resolve startup values from the canonical
`geko_defaults.defaults_for_parameters` instead of hard-coded dicts containing
the invalid `geko_cwall`. Adds build-time name validation in FD's
`test_config`. Only syntax-checked so far (no venv with numpy/skopt on the
writing machine). **Run `TESTING_optimizer_defaults.md` on this machine.**
That file also contains full reversal instructions.

### 3.3 Benchmark hyperparameters (changed by the user, plots regenerated)

N=24 (1-D) / N=48 (2-D); `random_state=42` for all optimizers;
`n_initial=9` (1-D) / `15` (2-D) for all optimizers with an init phase;
BO-first hybrids: BO phase 17 evals (1-D) / 32 (2-D). PSO: 4 particles,
derived 5 / 11 swarm iterations. FD: step_size=0.05, lr=0.015 standalone;
0.03 / 0.009 in BO->FD.

Best values currently shown in the figures (also in the report's Table 2;
true optima: local f=-1.8197 at x=0.8115, global f=-7.9167 at x=3.1915):

| Optimizer | 1-D best | 2-D best |
|---|---|---|
| BO        | x=3.191, f=-7.91673 | (3.19, 0.50), f=-7.91620 |
| NM        | x=0.812, f=-1.81971 | (0.81, 0.50), f=-1.81931 |
| FD        | x=0.809, f=-1.81966 | (0.79, 0.50), f=-1.81570 |
| PSO       | x=3.169, f=-7.90381 | (3.19, 0.50), f=-7.91529 |
| NM->BO    | x=0.812, f=-1.81971 | (3.18, 0.44), f=-7.73938 |
| FD->BO    | x=0.812, f=-1.81971 | (~3.19, ~0.55), f=-7.77858 |
| BO->NM    | x=3.191, f=-7.91671 | (3.19, 0.50), f=-7.91626 |
| BO->FD    | x=3.191, f=-7.91671 | (3.20, 0.50), f=-7.90393 |

### 3.4 Markdown files updated (mirroring the report fixes)

- `src/geko_bayesopt/optimizers_readme.md`: canonical defaults, boundary
  phrasing with the Fluent remark, max_iter remark removed.
- `src/geko_bayesopt/optimizers_documentation.md`: canonical defaults and the
  ValueError edge case, historical framings removed, max_iter remark removed.
- `optimizer_visualization/benchmark_readme.md`: budgets/splits/PSO iteration
  counts for N=24/48, FD visible-dot counts (12 of 24, 16 of 48), Sobol counts
  9/15, boundary section rewritten, corrected Lipschitz constant L ~ 52
  (verified numerically, max |f''| = 52.45 on [0.5, 3.5]; lr 0.015 < 1/L ~
  0.019; an older claim of L ~ 67 was wrong).

## 4. Known open issues in `_benchmark_core.py` (flagged, NOT fixed, user not yet decided)

1. **Stale `phase_split` entries in the `OPTIMIZERS` catalogue**: still 10/16
   (NM->BO), 9/16 (FD->BO), 10/20 (both BO-first hybrids), but the new
   `n_initial` values are 9/15, 9/15, 17/32, 17/32. The circle/triangle
   phase markers and "Phase 1 (N evals)" legends in the current hybrid
   figures are therefore wrong.
2. **BO->NM 2-D has `n_initial_sobol: 5`**, while BO->FD 2-D uses 15; likely
   an oversight given the user's stated goal of uniform BO initialization.
3. The comment block above the catalogue still documents the old 20/36
   budgets.

If these get fixed and the plots rerun, re-read the figure annotations and
update the report's Section 4 discussions and Table 2 (mainly the two
off-ridge NM->BO / FD->BO 2-D values).

## 5. Git state at hand-off (all uncommitted)

Modified: `src/geko_bayesopt/optimizer.py`, both optimizer .md files,
`optimizer_visualization/benchmark_readme.md`,
`optimizer_visualization/_benchmark_core.py` (user's own hyperparameter
changes), all eight PNGs (user's rerun).
Untracked: `optimizer_visualization/optimizer_report.tex`,
`TESTING_optimizer_defaults.md`, this file.

## 6. Suggested order of work on the new machine

1. Run the test plan in `TESTING_optimizer_defaults.md`.
2. Decide on and fix the three `_benchmark_core.py` issues (Section 4), rerun
   `benchmark_individual.py`, update the report numbers if trajectories moved.
3. Merge `optimizer_comparison_report.tex` into `optimizer_report.tex`
   (Section 1 of this file; clarify bibliography placeholders and the pending
   5D section with the user first).
4. Commit.
