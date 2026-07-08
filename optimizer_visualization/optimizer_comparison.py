"""
Optimizer comparison for the periodic hills Re=2800 case.

Usage
-----
    python optimizer_visualization\optimizer_comparison.py                # 1-D results (default)
    python optimizer_visualization\optimizer_comparison.py --dim 2d       # 2-D results
    python optimizer_visualization\optimizer_comparison.py --dim 5d       # 5-D results (BO vs PSO only)

Missing experiment folders and early-stopped runs are silently skipped.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).parent.parent.resolve()   # project root

# ------------------------------------------------------------------ #
# Configuration                                                       #
# ------------------------------------------------------------------ #

# Number of initial random-exploration samples in the BO run; vertical lines
# are drawn after these many iterations to mark where BO sampling stops and
# where BO hands off to NM/FD. Values differ by dimension because the BO
# hybrid configs use different n_initial_sobol / n_initial per dimension
# (see configs/optimizer_comparison_configs/{1D,2D}/07_*.json, 08_*.json).
_BO_SAMPLING_STOP = {"1d": 7, "2d": 13}
_BO_SWAP_ITER = {"1d": 14, "2d": 35}
_5D_CUTOFF_ONE = 50
_5D_CUTOFF_TWO = 100
_5D_CUTOFF_THREE = 200

# Shared evaluation budget of the seven-strategy comparison. The recorded BO
# runs are longer (45 evals in 1-D, 80 in 2-D) because BO is also compared
# against the PSO sweep; in the main comparison plots every run is truncated
# to this budget so BO's extra iterations only appear in the vs_pso plots.
_SHARED_BUDGET = {"1d": 21, "2d": 60}

# Main comparison: BO, NM, FD, and all hybrids. PSO is excluded here and
# plotted separately (see RUNS_1D_BO_VS_PSO / RUNS_2D_BO_VS_PSO below) since
# it uses a different evaluation budget and, in 1-D, a particle-count sweep
# instead of a single run.
RUNS_1D_BO = [
    ("BO_1D_ph2800",    "Bayesian Opt (GP)"),
    ("BO_NM_1D_ph2800", "Hybrid BO -> NM"),
    ("BO_FD_1D_ph2800", "Hybrid BO -> FD"),
    ("FD_1D_ph2800",    "Finite Differences"),
    ("FD_BO_1D_ph2800", "Hybrid FD -> BO"),
    ("NM_1D_ph2800",    "Nelder-Mead"),
    ("NM_BO_1D_ph2800", "Hybrid NM -> BO"),
]

RUNS_1D_BO_VS_PSO = [
    ("BO_1D_ph2800",     "Bayesian Opt (GP)"),
    ("PSO_1D_ph2800_p3", "Particle Swarm; 3 particles"),
    ("PSO_1D_ph2800_p5", "Particle Swarm; 5 particles"),
    ("PSO_1D_ph2800_p7", "Particle Swarm; 7 particles"),
    ("PSO_1D_ph2800_p9", "Particle Swarm; 9 particles"),
]

RUNS_2D_BO = [
    ("BO_2D_ph2800",    "Bayesian Opt (GP)"),
    ("BO_NM_2D_ph2800", "Hybrid BO -> NM"),
    ("BO_FD_2D_ph2800", "Hybrid BO -> FD"),
    ("FD_2D_ph2800",    "Finite Differences"),
    ("FD_BO_2D_ph2800", "Hybrid FD -> BO"),
    ("NM_2D_ph2800",    "Nelder-Mead"),
    ("NM_BO_2D_ph2800", "Hybrid NM -> BO"),
]

RUNS_2D_BO_VS_PSO = [
    ("BO_2D_ph2800",        "Bayesian Opt (GP)"),
    ("PSO_2D_ph2800_p10",   "Particle Swarm; 10 particles"),
    ("PSO_2D_ph2800_p15",   "Particle Swarm; 15 particles"),
    ("PSO_2D_ph2800_p20",   "Particle Swarm; 20 particles"),
]

RUNS_5D_BO_VS_PSO = [
    ("BO_5D_ph2800",        "Bayesian Opt (GP)"),
    ("PSO_5D_ph2800_p15",   "Particle Swarm; 15 particles"),
    ("PSO_5D_ph2800_p30",   "Particle Swarm; 30 particles"),
    ("PSO_5D_ph2800_p50",   "Particle Swarm; 50 particles")
]


_TITLES = {
    "1d": "Optimizer Comparison - Periodic Hills Re=2800, 1D (geko_csep)",
    "2d": "Optimizer Comparison - Periodic Hills Re=2800, 2D (geko_csep, geko_cnw)",
    # Parameter list on its own line: the full 5-D title is too wide for the
    # 10x5 figure and gets clipped otherwise.
    "5d": "Optimizer Comparison - Periodic Hills Re=2800, 5D\n(geko_csep, geko_cnw, geko_cmix, geko_cturb, geko_cjet)"
}

_SUBDIRS = {
    "1d": "one-param-runs",
    "2d": "two-param-runs",
    "5d": "five-param-runs",
}

# metadata.csv parameter columns to report the argmin for, and how to display
# them in the legend (e.g. "; Csep=0.886" or "; Csep=0.886, Cnw=0.50").
_PARAM_COLS = {
    "1d": ["geko_csep"],
    "2d": ["geko_csep", "geko_cnw"],
    "5d": ["geko_csep", "geko_cnw", "geko_cmix", "geko_cturb", "geko_cjet"]
}
_PARAM_DISPLAY_NAMES = {
    "geko_csep": "Csep",
    "geko_cnw": "Cnw",
    "geko_cmix": "Cmix",
    "geko_cturb": "Cturb",
    "geko_cjet": "Cjet"
}

# Iterations <= this value are excluded from the "after_iter<N>" plot
# (stage 1 -- shortly after BO sampling stops).
_CUT_ITER_1D = _BO_SAMPLING_STOP["1d"]
_CUT_ITER_2D = _BO_SAMPLING_STOP["2d"]
_CUT_ITER_5D = _5D_CUTOFF_ONE
# Iterations <= this value are excluded from the "after_iter<N>" plot
# (stage 2 -- everything after the BO -> NM/FD swap). Reuses _BO_SWAP_ITER
# since that is exactly where the second vertical line sits.
_CUT_ITER_1D_STAGE2 = _BO_SWAP_ITER["1d"]
_CUT_ITER_2D_STAGE2 = _BO_SWAP_ITER["2d"]
_CUT_ITER_5D_STAGE2 = _5D_CUTOFF_TWO

_CUT_ITER_5D_STAGE3 =  _5D_CUTOFF_THREE


# Second cutoff for the dedicated BO-vs-PSO plots (the first cutoff reuses
# _BO_SAMPLING_STOP directly, since that is exactly where BO's own sampling
# phase ends). Chosen well into the PSO runs' swarm iterations while leaving
# enough tail evaluations to compare late-stage convergence.
_CUT_ITER_1D_PSO = 28
_CUT_ITER_2D_PSO = 51

_PLOTS_ROOT = Path(__file__).parent / "plots" / "comparison"


def _plots_dir(dim: str) -> Path:
    """Per-dimension output folder, e.g. plots/comparison/1d/."""
    d = _PLOTS_ROOT / dim
    d.mkdir(parents=True, exist_ok=True)
    return d

# Fixed label -> color mapping so each optimizer keeps the same colour across
# every plot it appears in (RUNS_1D_BO and RUNS_2D_BO share the same label
# order, so one mapping covers both dimensions). Each optimizer family gets
# its own hue, with hybrids taking a lighter shade of whichever optimizer
# drives the second phase: BO family in blue, FD family in black/grey, NM
# family in orange. PSO (BO-vs-PSO plots only) reuses BO's dark blue and
# gives each particle-count run its own shade of orange.
_RUN_COLORS = {
    "Bayesian Opt (GP)":  "#08306b",  # dark blue
    "Hybrid BO -> NM":    "#3182bd",  # blue
    "Hybrid BO -> FD":    "#9ecae1",  # light blue
    "Finite Differences": "#000000",  # black
    "Hybrid FD -> BO":    "#636363",  # light black (dark grey)
    "Nelder-Mead":        "#e6550d",  # orange
    "Hybrid NM -> BO":    "#fdae6b",  # light orange

    "Particle Swarm; 3 particles":  "#fdd0a2",
    "Particle Swarm; 5 particles":  "#fdae6b",
    "Particle Swarm; 7 particles":  "#f16913",
    "Particle Swarm; 9 particles":  "#a63603",
    "Particle Swarm; 10 particles": "#fdae6b",
    "Particle Swarm; 15 particles": "#f16913",
    "Particle Swarm; 20 particles": "#a63603",

    "Particle Swarm; 15 particles, 5D": "#fdae6b",
    "Particle Swarm; 30 particles, 5D": "#f16913",
    "Particle Swarm; 50 particles, 5D": "#a63603",
}


# ------------------------------------------------------------------ #
# Loading results                                                     #
# ------------------------------------------------------------------ #

def _load_scores(experiment_id: str, dim: str) -> tuple[np.ndarray, list[dict[str, float]]] | None:
    """Return (scores, param_rows) from metadata.csv, or None if absent.

    Only trial_role == "optimizer" rows are returned: metadata.csv starts with
    a baseline row (GEKO-default run) that is not an optimizer iteration --
    including it would shift every curve by one and misalign the phase markers.

    param_rows[i] holds the parameter values (see _PARAM_COLS) for scores[i],
    used to report the argmin parameter values in the legend.
    """
    csv_path = (
        REPO_ROOT
        / "results" / "experiments" / "optimizer_comparison"
        / _SUBDIRS[dim]
        / experiment_id
        / "metadata.csv"
    )
    if not csv_path.exists():
        return None
    cols = _PARAM_COLS[dim]
    scores = []
    param_rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["trial_role"] != "optimizer":
                continue
            scores.append(float(row["score"]))
            param_rows.append({c: float(row[c]) for c in cols})
    return (np.array(scores), param_rows) if scores else None


def _argmin_suffix(scores: np.ndarray, param_rows: list[dict[str, float]], dim: str) -> str:
    """'; Csep=0.886; cost=1.234' -- parameters and cost of the best point found."""
    best_idx = int(np.argmin(scores))
    best = param_rows[best_idx]
    parts = [f"{_PARAM_DISPLAY_NAMES[c]}={best[c]:.3f}" for c in _PARAM_COLS[dim]]
    return "; " + ", ".join(parts) + f"; cost={scores[best_idx]:.4f}"


def _load_runs(
    runs: list[tuple[str, str]], dim: str, max_iter_cap: int | None = None
) -> list[tuple[str, str, np.ndarray]]:
    """Load scores for each (experiment_id, label) pair, skipping missing ones.

    Returns (base_label, display_label, scores) tuples: base_label is used to
    look up a stable colour, display_label (base_label + argmin suffix) is
    what's shown in the legend.

    max_iter_cap truncates every run to its first ``max_iter_cap`` evaluations
    (used to cut BO's extended history down to the shared budget in the main
    comparison plots); the legend's argmin then also refers to the truncated
    history.
    """
    loaded: list[tuple[str, str, np.ndarray]] = []
    for exp_id, label in runs:
        result = _load_scores(exp_id, dim)
        if result is None:
            print(f"[plot] No results yet for '{label}' -- skipping.")
            continue
        scores, param_rows = result
        if max_iter_cap is not None:
            scores, param_rows = scores[:max_iter_cap], param_rows[:max_iter_cap]
        display_label = label + _argmin_suffix(scores, param_rows, dim)
        loaded.append((label, display_label, scores))
    if not loaded:
        print("[plot] No real results found -- nothing to plot.")
    return loaded


# ------------------------------------------------------------------ #
# Plotting helpers                                                    #
# ------------------------------------------------------------------ #

def _set_adaptive_ylim(ax, bsf_arrays: list[np.ndarray], pad_frac: float = 0.05) -> None:
    """Zoom y-axis to the actual data range with proportional padding."""
    all_y = np.concatenate(bsf_arrays)
    y_min, y_max = all_y.min(), all_y.max()
    pad = (y_max - y_min) * pad_frac
    ax.set_ylim(y_min - pad, y_max + pad)


def _run_color(base_label: str, dim: str) -> str | None:
    """Colour for a run label; 5-D PSO runs have their own entries (', 5D' keys)."""
    if dim == "5d":
        return _RUN_COLORS.get(f"{base_label}, 5D", _RUN_COLORS.get(base_label))
    return _RUN_COLORS.get(base_label)


def _draw_phase_markers(ax, dim: str, cut_iter: int | None = None, show_swap_marker: bool = True) -> None:
    """Draw the BO-phase vertical markers (sampling stop / BO->NM/FD swap)."""
    if dim not in _BO_SAMPLING_STOP:   # 5-D runs have no phase markers
        return
    sampling_stop = _BO_SAMPLING_STOP[dim]
    swap_iter = _BO_SWAP_ITER[dim]
    if cut_iter is None or sampling_stop > cut_iter:
        ax.axvline(x=sampling_stop, color="black", linestyle="--", linewidth=1.2,
                   alpha=0.5, label=f"BO sampling stops (iter {sampling_stop})")
    if show_swap_marker and (cut_iter is None or swap_iter > cut_iter):
        ax.axvline(x=swap_iter, color="dimgray", linestyle="--", linewidth=1.2,
                   alpha=0.5, label=f"BO swaps to NM/FD (iter {swap_iter})")


# ------------------------------------------------------------------ #
# Plot functions                                                      #
# ------------------------------------------------------------------ #

def plot_comparison_full(dim: str, runs: list[tuple[str, str]], slug: str = "",
                          show_swap_marker: bool = True,
                          max_iter_cap: int | None = None) -> None:
    """Plot every iteration, uncut (up to max_iter_cap if given)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    loaded = _load_runs(runs, dim, max_iter_cap=max_iter_cap)

    bsf_arrays: list[np.ndarray] = []
    for base_label, display_label, scores in loaded:
        best_so_far = np.minimum.accumulate(scores)
        iters = np.arange(1, len(scores) + 1)
        ax.plot(iters, best_so_far, linewidth=2, label=display_label, color=_run_color(base_label, dim))
        bsf_arrays.append(best_so_far)

    _draw_phase_markers(ax, dim, show_swap_marker=show_swap_marker)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best cost so far")
    ax.set_title(_TITLES[dim])
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    if bsf_arrays:
        _set_adaptive_ylim(ax, bsf_arrays)

    fig.tight_layout()
    out_path = _plots_dir(dim) / f"optimizer_comparison_{dim}_{slug}full.png"
    fig.savefig(out_path, dpi=150)
    print(f"[plot] Saved -> {out_path}")
    plt.close(fig)


def plot_comparison_cut(dim: str, runs: list[tuple[str, str]], cut_iter: int, slug: str = "",
                         show_swap_marker: bool = True, title: str | None = None,
                         max_iter_cap: int | None = None) -> None:
    """Same as plot_comparison_full but omits all iterations <= cut_iter."""
    fig, ax = plt.subplots(figsize=(10, 5))
    loaded = _load_runs(runs, dim, max_iter_cap=max_iter_cap)

    max_iters = max(len(s) for _, _, s in loaded) if loaded else 0

    bsf_arrays: list[np.ndarray] = []
    for base_label, display_label, scores in loaded:
        best_so_far = np.minimum.accumulate(scores)
        iters = np.arange(1, len(scores) + 1)

        mask = iters > cut_iter
        iters_cut = iters[mask]
        bsf_cut = best_so_far[mask]

        if len(iters_cut) == 0:
            continue

        ax.plot(iters_cut, bsf_cut, linewidth=2, label=display_label, color=_run_color(base_label, dim))
        bsf_arrays.append(bsf_cut)

    _draw_phase_markers(ax, dim, cut_iter=cut_iter, show_swap_marker=show_swap_marker)

    ax.set_xlim(cut_iter + 0.5, max_iters + 0.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best cost so far")
    ax.set_title((title or _TITLES[dim]) + f" (from iter {cut_iter + 1})")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    if bsf_arrays:
        _set_adaptive_ylim(ax, bsf_arrays)

    fig.tight_layout()
    out_path = _plots_dir(dim) / f"optimizer_comparison_{dim}_{slug}after_iter{cut_iter}.png"
    fig.savefig(out_path, dpi=150)
    print(f"[plot] Saved -> {out_path}")
    plt.close(fig)


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot optimizer comparison.")
    parser.add_argument(
        "--dim", choices=["1d", "2d", "5d"], default="1d",
        help="Which parameter dimension to plot (default: 1d).",
    )
    args = parser.parse_args()
    dim = args.dim

    if dim == "5d":
        # 5-D final comparison: BO vs the PSO particle-count sweep only (no
        # NM/FD runs, hence no phase markers). One uncut full plot plus three
        # after-cut plots. BO is kept in every plot, including the third cut.
        plot_comparison_full(dim, RUNS_5D_BO_VS_PSO, slug="vs_pso_",
                             show_swap_marker=False)
        for cut in (_CUT_ITER_5D, _CUT_ITER_5D_STAGE2, _CUT_ITER_5D_STAGE3):
            plot_comparison_cut(dim, RUNS_5D_BO_VS_PSO, cut_iter=cut,
                                slug="vs_pso_", show_swap_marker=False)
        raise SystemExit(0)

    runs_bo = RUNS_1D_BO if dim == "1d" else RUNS_2D_BO
    runs_pso = RUNS_1D_BO_VS_PSO if dim == "1d" else RUNS_2D_BO_VS_PSO

    cut_iter_stage1 = _CUT_ITER_1D if dim == "1d" else _CUT_ITER_2D
    cut_iter_stage2 = _CUT_ITER_1D_STAGE2 if dim == "1d" else _CUT_ITER_2D_STAGE2
    cut_iter_pso = _CUT_ITER_1D_PSO if dim == "1d" else _CUT_ITER_2D_PSO

    # Main comparison: BO, NM, FD, and all hybrids. Every run is truncated to
    # the shared budget: BO's recorded history is longer (it is extended for
    # the PSO comparison below) and those extra iterations are not meant to
    # be shown here.
    cap = _SHARED_BUDGET[dim]
    plot_comparison_full(dim, runs_bo, max_iter_cap=cap)
    plot_comparison_cut(dim, runs_bo, cut_iter=cut_iter_stage1, max_iter_cap=cap)
    plot_comparison_cut(dim, runs_bo, cut_iter=cut_iter_stage2, max_iter_cap=cap)

    # Dedicated BO-vs-PSO comparison: no NM/FD runs here, so the
    # "BO swaps to NM/FD" marker doesn't apply. BO's full extended history is
    # intended here, so no cap: it stays visible through every cutoff.
    plot_comparison_full(dim, runs_pso, slug="vs_pso_", show_swap_marker=False)
    plot_comparison_cut(dim, runs_pso, cut_iter=_BO_SAMPLING_STOP[dim], slug="vs_pso_", show_swap_marker=False)
    plot_comparison_cut(dim, runs_pso, cut_iter=cut_iter_pso, slug="vs_pso_", show_swap_marker=False)
