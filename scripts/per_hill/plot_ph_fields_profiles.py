"""Periodic-hill plotting script for DNS, optimized simulation, and optional GEKO default comparison.

Run from the repository root and pass the plotting config explicitly:

     python scripts/per_hill/plot_ph_fields_profiles.py scripts/per_hill/plots/<config-name>.json

This script is controlled by an external JSON configuration file. The JSON defines
the case name, the simulation file, the DNS file, the column indices for x, y, and
each plotted field, the optional output directory, as well as the plotting options.


For each selected field, the script can generate:
    1. comparison plots between DNS and the optimized simulation;
    2. comparison plots between the fixed GEKO default solution and the optimized simulation;
    3. error plots, where the reference field is interpolated onto the
       simulation points before computing simulation - reference;
    4. optional vertical profiles at the x-locations specified in the JSON.


Relevant JSON plotting switches:
    make_profiles:
        If true, profile plots are generated.
        If false, only field comparison plots are generated.

    compare_default:
        If true, the optimized simulation is also compared against the GEKO default file.
        If false, only the DNS comparison is generated.

    normalize_error:
        If true (default), error panels show (simulation - reference) / std(reference)
        so fields with different units/magnitudes can share a meaningful scale.

    mask_hill:
        If true, points below the periodic-hill surface are hidden.

Ex.
{
  "name": "periodic_hills_2800_v1",
  "output_dir": "results/periodic_hills/periodic_hills_2800/periodic_hills_2800_csep_cnw_cmix_cturb/plots",
  "simulation": {
    "path": "results/fluent/periodic_hills_2800_v1/alpha1.0_Re2800_Csep0.8792_Cnw0.4893_Cmix0.1918_Cjet0.8705_Cturb1.9724.ascii",
    "x": 1,
    "y": 2,
    "fields": {
      "Ux": 8,
      "Uy": 7,
      "turb-kinetic-energy": 5,
      "production-of-k": 4,
      "dissipation": 3
    }
  },
  "dns": {
    "path": "data/dns/periodic_hills/pehill-2800-Re-DNS/dns_avg_Re2800_columnwise_organized.ascii",
    "x": 1,
    "y": 2,
    "fields": {
      "Ux": 3,
      "Uy": 4,
      "turb-kinetic-energy": 10,
      "production-of-k": 7,
      "dissipation": 8
    }
  },
  "plots": {
    "fields": [
      "Ux",
      "Uy",
      "turb-kinetic-energy",
      "production-of-k",
      "dissipation"
    ],
    "x_locations": [
      2.0,
      4.5,
      8.0
    ],
    "x_tol": 0.1,
    "make_profiles": true,
    "compare_default": true,
    "normalize_error": true,
    "mask_hill": true
  }
}

"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from geko_bayesopt.objective.field_error import hill_surface

DEFAULT_CONFIG_PATH = os.path.join(
    REPO_ROOT, "scripts", "per_hill", "plots", "plot_config_2800.json"
)



def repo_path(path):
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def load_config(config_path):
    with open(repo_path(config_path), "r", encoding="utf-8") as f:
        return json.load(f)


def read_table(path):
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        first_line = fh.readline()

    if path.suffix.lower() == ".csv" or "," in first_line:
        return pd.read_csv(path, skipinitialspace=True)
    return pd.read_csv(path, sep=r"\s+", engine="python", skipinitialspace=True)


def get_column(df, col):
    if isinstance(col, int):
        return df.iloc[:, col].to_numpy(dtype=float)
    return df[col].to_numpy(dtype=float)


def hill_keep_mask(x, y, domain_length=9.0):
    y_surf = hill_surface(np.asarray(x, dtype=float), domain_length)
    return np.asarray(y, dtype=float) >= (y_surf - 1e-6)


def apply_hill_mask(data, *, domain_length=9.0):
    keep = hill_keep_mask(data["x"], data["y"], domain_length)
    return {
        name: values[keep] if len(values) == len(keep) else values
        for name, values in data.items()
    }


def plot_field(ax, x, y, values, cmap='viridis', vmin=None, vmax=None):
    triang = tri.Triangulation(x, y)

    # Mask triangles with long edges — these are usually the ones crossing the hill geometry
    tris = triang.triangles

    edge_01 = np.sqrt(
        (x[tris[:, 0]] - x[tris[:, 1]])**2
        + (y[tris[:, 0]] - y[tris[:, 1]])**2
    )
    edge_12 = np.sqrt(
        (x[tris[:, 1]] - x[tris[:, 2]])**2
        + (y[tris[:, 1]] - y[tris[:, 2]])**2
    )
    edge_20 = np.sqrt(
        (x[tris[:, 2]] - x[tris[:, 0]])**2
        + (y[tris[:, 2]] - y[tris[:, 0]])**2
    )

    max_edge = np.maximum.reduce([edge_01, edge_12, edge_20])

    triang.set_mask(max_edge > 0.15)

    levels = np.linspace(vmin, vmax, 51) if (vmin is not None and vmax is not None) else 50
    tcf = ax.tricontourf(triang, values, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    return tcf


def save_field_plot(x, y, values, title, label, filepath, cmap='viridis', vmin=None, vmax=None):
    """Renderiza um campo 2D e guarda em ficheiro."""
    fig, ax = plt.subplots(figsize=(10, 6))
    tcf = plot_field(ax, x, y, values, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(tcf, ax=ax, label=label)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(True)
    fig.savefig(filepath)
    plt.close(fig)


def load_dns_data(dns_cfg):
    """Carrega e devolve os campos do ficheiro DNS."""
    df = read_table(repo_path(dns_cfg["path"]))

    x = get_column(df, dns_cfg["x"])
    y = get_column(df, dns_cfg["y"])

    out = {"x": x, "y": y}

    for field_name, col in dns_cfg["fields"].items():
        out[field_name] = get_column(df, col)

    return out


def load_sim_data(sim_cfg):
    """Carrega e devolve os campos do ficheiro de simulação."""
    df = read_table(repo_path(sim_cfg["path"]))

    x = get_column(df, sim_cfg["x"])
    y = get_column(df, sim_cfg["y"])

    out = {"x": x, "y": y}

    for field_name, col in sim_cfg["fields"].items():
        out[field_name] = get_column(df, col)

    return out


def build_default_cfg(cfg, sim_cfg, compare_default):
    if not compare_default:
        return None

    default_cfg = cfg.get("default")
    if default_cfg is None or not default_cfg.get("path"):
        raise ValueError(
            "plots.compare_default is true, so the JSON must define "
            'a top-level "default" block with a "path" entry.'
        )

    merged = {
        "x": sim_cfg["x"],
        "y": sim_cfg["y"],
        "fields": sim_cfg["fields"],
    }
    merged.update(default_cfg)
    return merged


def interpolate_reference_to_sim_grid(ref_x, ref_y, ref_values, sim_x, sim_y):
    """Interpolate reference values to simulation coordinates for error only."""
    points = np.column_stack((ref_x, ref_y))
    target = np.column_stack((sim_x, sim_y))

    ref_on_sim = griddata(points, ref_values, target, method="linear")
    if np.any(~np.isfinite(ref_on_sim)):
        nearest = griddata(points, ref_values, target, method="nearest")
        ref_on_sim = np.where(np.isfinite(ref_on_sim), ref_on_sim, nearest)
    return ref_on_sim


def _safe_std(values):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    std = float(np.std(finite))
    return max(std, 1e-8)


def calculate_error_field(ref, sim, field, *, normalize=True):
    """Return simulation-reference error on the simulation grid.

    When normalize=True, the signed error is divided by the standard
    deviation of the interpolated reference field. This mirrors the field
    objective's DNS-standard-deviation normalization and makes different
    fields comparable on one color scale.
    """
    ref_on_sim = interpolate_reference_to_sim_grid(
        ref["x"], ref["y"], ref[field],
        sim["x"], sim["y"],
    )
    error = sim[field] - ref_on_sim
    if normalize:
        error = error / _safe_std(ref_on_sim)
    return error


def common_error_limits_by_field(comparisons, fields, *, normalize=True):
    """Find one symmetric color limit per field across requested comparisons."""
    limits = {}
    for field in fields:
        maxima = []
        for ref, sim in comparisons:
            error = calculate_error_field(ref, sim, field, normalize=normalize)
            finite_abs = np.abs(error[np.isfinite(error)])
            if finite_abs.size:
                maxima.append(float(np.max(finite_abs)))

        if not maxima:
            limits[field] = 1e-12
            continue
        limit = max(maxima)
        limits[field] = limit if np.isfinite(limit) and limit > 0.0 else 1e-12
    return limits


def save_error_plot(
    ref_label,
    ref,
    sim,
    field,
    filepath,
    *,
    error_limit=None,
    normalize=True,
):
    """Calculate error using interpolation, then plot with old plot_field()."""
    error = calculate_error_field(ref, sim, field, normalize=normalize)

    emax = error_limit
    if emax is None:
        finite_abs = np.abs(error[np.isfinite(error)])
        emax = float(np.max(finite_abs)) if finite_abs.size else 1e-12
    if not np.isfinite(emax) or emax == 0.0:
        emax = 1e-12

    label = (
        f"{field} normalized difference"
        if normalize
        else f"{field} difference"
    )
    save_field_plot(
        sim["x"],
        sim["y"],
        error,
        title=f"{ref_label} {field}",
        label=label,
        filepath=filepath,
        cmap="coolwarm",
        vmin=-emax,
        vmax=emax,
    )


def stitch_images(image_paths, output_path, layout="horizontal"):
    """Stitch already-rendered old-style PNGs into one comparison image."""
    from PIL import Image, ImageOps

    images = [Image.open(path).convert("RGB") for path in image_paths]

    if layout == "horizontal":
        max_h = max(img.height for img in images)
        padded = [
            ImageOps.expand(img, border=(0, 0, 0, max_h - img.height), fill="white")
            for img in images
        ]
        total_w = sum(img.width for img in padded)
        canvas = Image.new("RGB", (total_w, max_h), "white")
        x = 0
        for img in padded:
            canvas.paste(img, (x, 0))
            x += img.width
    else:
        max_w = max(img.width for img in images)
        padded = [
            ImageOps.expand(img, border=(0, 0, max_w - img.width, 0), fill="white")
            for img in images
        ]
        total_h = sum(img.height for img in padded)
        canvas = Image.new("RGB", (max_w, total_h), "white")
        y = 0
        for img in padded:
            canvas.paste(img, (0, y))
            y += img.height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def shared_field_limits(fields, ref, sim, default=None):
    """Return one shared min/max per field across the available datasets."""
    limits = {}
    for field in fields:
        values = [ref[field], sim[field]]
        if default is not None:
            values.append(default[field])

        all_values = []
        for arr in values:
            arr = np.asarray(arr, dtype=float)
            all_values.append(arr[np.isfinite(arr)])

        combined = np.concatenate(all_values) if all_values else np.array([])
        if combined.size == 0:
            limits[field] = (0.0, 1.0)
        else:
            limits[field] = (float(combined.min()), float(combined.max()))
    return limits


def save_comparison_list(
    *,
    case_name,
    field,
    ref_label,
    sim_label,
    ref,
    sim,
    output_path,
    layout="horizontal",
    vmin=None,
    vmax=None,
    error_limit=None,
    normalize_error=True,
):
    
    if vmin is None:
        vmin = min(ref[field].min(), sim[field].min())
    if vmax is None:
        vmax = max(ref[field].max(), sim[field].max())

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        image_paths = []

        ref_path = tmp / f"01_{ref_label}_{field}.png"
        save_field_plot(
            ref["x"], ref["y"], ref[field],
            title=f"DNS {field}",
            label=field,
            filepath=ref_path,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        image_paths.append(ref_path)

        sim_path = tmp / f"02_simulation_{field}.png"
        save_field_plot(
            sim["x"], sim["y"], sim[field],
            title=f"{sim_label} {field}",
            label=field,
            filepath=sim_path,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        image_paths.append(sim_path)

        err_path = tmp / f"03_error_{field}.png"
        save_error_plot(
            f"Error {sim_label} - DNS",
            ref,
            sim,
            field,
            err_path,
            error_limit=error_limit,
            normalize=normalize_error,
        )
        image_paths.append(err_path)

        stitch_images(image_paths, output_path, layout=layout)



def _repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path(REPO_ROOT) / path


def _read_table(path: Path) -> pd.DataFrame:
    return read_table(path)


def _col(df: pd.DataFrame, col):
    if isinstance(col, int):
        return df.iloc[:, col]
    return df[col]


def _apply_hill_mask_frame(
    df: pd.DataFrame,
    x_col,
    y_col,
    *,
    domain_length=9.0,
) -> pd.DataFrame:
    keep = hill_keep_mask(_col(df, x_col), _col(df, y_col), domain_length)
    return df.loc[keep].copy()


def load_profile_data(sim_cfg: dict, dns_cfg: dict, default_cfg: dict | None = None):
    sim = _read_table(_repo_path(sim_cfg["path"]))
    dns = _read_table(_repo_path(dns_cfg["path"]))
    default = None
    if default_cfg is not None:
        default = _read_table(_repo_path(default_cfg["path"]))

    return sim, dns, default


def plot_profiles(
    x_locations: list[float],
    fields: list[str],
    sim: pd.DataFrame,
    dns: pd.DataFrame,
    sim_cfg: dict,
    dns_cfg: dict,
    output_dir: Path,
    tol: float,
    default: pd.DataFrame | None = None,
    default_cfg: dict | None = None,
    case_name: str | None = None,
) -> None:
    """
    Plot horizontal profiles (field vs y) at each x location for each field.
    Saves one image per field.
    """
    unknown_sim = [f for f in fields if f not in sim_cfg["fields"]]
    unknown_dns = [f for f in fields if f not in dns_cfg["fields"]]
    if unknown_sim:
        raise ValueError(f"Unknown simulation fields: {unknown_sim}. Choose from {list(sim_cfg['fields'])}")
    if unknown_dns:
        raise ValueError(f"Unknown DNS fields: {unknown_dns}. Choose from {list(dns_cfg['fields'])}")

    n_x = len(x_locations)

    sim_x_col = sim_cfg["x"]
    sim_y_col = sim_cfg["y"]
    dns_x_col = dns_cfg["x"]
    dns_y_col = dns_cfg["y"]

    for field in fields:
        sim_col = sim_cfg["fields"][field]
        dns_col = dns_cfg["fields"][field]

        fig, axes = plt.subplots(1, n_x, figsize=(5 * n_x, 4), squeeze=False)

        for col, x_val in enumerate(x_locations):
            ax = axes[0][col]

            sim_slice = (
                sim[np.abs(_col(sim, sim_x_col) - x_val) < tol]
                .sort_values(by=sim_y_col if isinstance(sim_y_col, str) else sim.columns[sim_y_col])
            )
            dns_slice = (
                dns[np.abs(_col(dns, dns_x_col) - x_val) < tol]
                .sort_values(by=dns_y_col if isinstance(dns_y_col, str) else dns.columns[dns_y_col])
            )

            ax.plot(
                _col(sim_slice, sim_col),
                _col(sim_slice, sim_y_col),
                color="tab:red",
                linewidth=1.5,
                alpha=0.7,
                label="Optimized",
            )
            ax.plot(
                _col(dns_slice, dns_col),
                _col(dns_slice, dns_y_col),
                color="tab:blue",
                linewidth=1.5,
                alpha=0.7,
                label="DNS",
            )

            if default is not None and default_cfg is not None:
                default_x_col = default_cfg["x"]
                default_y_col = default_cfg["y"]
                default_col = default_cfg["fields"][field]
                default_slice = (
                    default[np.abs(_col(default, default_x_col) - x_val) < tol]
                    .sort_values(by=default_y_col if isinstance(default_y_col, str) else default.columns[default_y_col])
                )
                ax.plot(
                    _col(default_slice, default_col),
                    _col(default_slice, default_y_col),
                    color="tab:green",
                    linewidth=1.5,
                    alpha=0.7,
                    label="Default",
                )

            ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
            ax.set_xlabel(field)
            ax.set_ylabel("y")
            ax.set_title(f"{field} — x = {x_val}  (tol = {tol})")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        if case_name:
            out_path = output_dir / f"profiles_{case_name}_{field}.png"
        else:
            out_path = output_dir / f"{field}_profiles.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
        print(f"Saved: {out_path}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default=DEFAULT_CONFIG_PATH,
        help="Path to JSON config. Default: scripts/per_hill/plots/plot_config_2800.json",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    output_dir = cfg.get("output_dir")
    if output_dir is not None:
        output_directory = Path(repo_path(output_dir))
    else:
        results_dir = cfg.get("results_dir")
        if results_dir is None:
            case_name = cfg.get("name", cfg.get("experiment_id", "unknown"))
            results_dir = os.path.join("results", "experiments", case_name)
        output_directory = Path(repo_path(results_dir)) / "plots"

    output_directory.mkdir(parents=True, exist_ok=True)
    simulation_directory = output_directory / "simulation"
    default_directory = output_directory / "default"
    profiles_directory = output_directory / "profiles"
    simulation_directory.mkdir(parents=True, exist_ok=True)
    default_directory.mkdir(parents=True, exist_ok=True)
    profiles_directory.mkdir(parents=True, exist_ok=True)

    sim_cfg = cfg["simulation"]
    dns_cfg = cfg["dns"]
    plots_cfg = cfg["plots"]
    fields = plots_cfg["fields"]

    compare_default = bool(plots_cfg.get("compare_default", False))
    default_cfg = build_default_cfg(cfg, sim_cfg, compare_default)
    make_profiles = bool(plots_cfg.get("make_profiles", bool(plots_cfg.get("x_locations", []))))
    layout = plots_cfg.get("comparison_layout", "horizontal")
    mask_hill = bool(plots_cfg.get("mask_hill", cfg.get("mask_hill", False)))

    dns = load_dns_data(dns_cfg)
    sim = load_sim_data(sim_cfg)
    default = load_sim_data(default_cfg) if compare_default else None
    if mask_hill:
        dns = apply_hill_mask(dns)
        sim = apply_hill_mask(sim)
        if default is not None:
            default = apply_hill_mask(default)
    field_limits = shared_field_limits(fields, dns, sim, default)
    normalize_error = bool(plots_cfg.get("normalize_error", True))
    error_comparisons = [(dns, sim)]
    if compare_default and default is not None:
        error_comparisons.append((dns, default))
    error_limits = common_error_limits_by_field(
        error_comparisons,
        fields,
        normalize=normalize_error,
    )

    for field in fields:
        compare_path = simulation_directory / f"Sim_DNS_{cfg['name']}_{field}.png"
        lo, hi = field_limits[field]
        save_comparison_list(
            case_name=cfg["name"],
            field=field,
            ref_label="DNS",
            sim_label="Simulation",
            ref=dns,
            sim=sim,
            output_path=compare_path,
            layout=layout,
            vmin=lo,
            vmax=hi,
            error_limit=error_limits.get(field),
            normalize_error=normalize_error,
        )
        print(f"Saved: {compare_path}")

        if compare_default:
            compare_default_path = default_directory / f"Default_DNS_{cfg['name']}_{field}.png"
            save_comparison_list(
                case_name=cfg["name"],
                field=field,
                ref_label="DNS",
                sim_label="Default",
                ref=dns,
                sim=default,
                output_path=compare_default_path,
                layout=layout,
                vmin=lo,
                vmax=hi,
                error_limit=error_limits.get(field),
                normalize_error=normalize_error,
            )
            print(f"Saved: {compare_default_path}")

    if make_profiles:
        sim_df, dns_df, default_df = load_profile_data(
            sim_cfg,
            dns_cfg,
            default_cfg if compare_default else None,
        )
        if mask_hill:
            sim_df = _apply_hill_mask_frame(
                sim_df,
                sim_cfg["x"],
                sim_cfg["y"],
            )
            dns_df = _apply_hill_mask_frame(
                dns_df,
                dns_cfg["x"],
                dns_cfg["y"],
            )
            if default_df is not None and default_cfg is not None:
                default_df = _apply_hill_mask_frame(
                    default_df,
                    default_cfg["x"],
                    default_cfg["y"],
                )
        plot_profiles(
            x_locations=plots_cfg["x_locations"],
            fields=fields,
            sim=sim_df,
            dns=dns_df,
            sim_cfg=sim_cfg,
            dns_cfg=dns_cfg,
            output_dir=profiles_directory,
            tol=plots_cfg["x_tol"],
            default=default_df if compare_default else None,
            default_cfg=default_cfg if compare_default else None,
            case_name=cfg["name"],
        )


if __name__ == "__main__":
    main()
