"""Plot periodic-hill DNS, optimized result, default result, and profiles.

Run from the repository root:

    python scripts/per_hill/plot_ph_fields_profiles.py scripts/per_hill/plots/<config>.json
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from geko_bayesopt.objective.field_error import hill_surface

DEFAULT_CONFIG_PATH = REPO_ROOT / "scripts" / "per_hill" / "plots" / "plot_config_2800.json"

AXIS_LABEL_SIZE = 12
TITLE_SIZE = 13
TICK_LABEL_SIZE = 11
LEGEND_SIZE = 10


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: str | Path) -> dict:
    with repo_path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def read_table(path: str | Path) -> pd.DataFrame:
    path = repo_path(path)
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        first_line = fh.readline()
    if path.suffix.lower() == ".csv" or "," in first_line:
        return pd.read_csv(path, skipinitialspace=True)
    return pd.read_csv(path, sep=r"\s+", engine="python", skipinitialspace=True)


def column(df: pd.DataFrame, col):
    return df.iloc[:, col] if isinstance(col, int) else df[col]


def values(df: pd.DataFrame, col) -> np.ndarray:
    return column(df, col).to_numpy(dtype=float)


def load_dataset(cfg: dict) -> dict[str, np.ndarray]:
    df = read_table(cfg["path"])
    data = {"x": values(df, cfg["x"]), "y": values(df, cfg["y"])}
    for name, col in cfg["fields"].items():
        data[name] = values(df, col)
    return data


def apply_hill_mask(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    keep = np.asarray(data["y"], dtype=float) >= (hill_surface(data["x"]) - 1e-6)
    return {
        name: arr[keep] if len(arr) == len(keep) else arr
        for name, arr in data.items()
    }


def mask_frame(df: pd.DataFrame, x_col, y_col) -> pd.DataFrame:
    keep = np.asarray(column(df, y_col), dtype=float) >= (
        hill_surface(np.asarray(column(df, x_col), dtype=float)) - 1e-6
    )
    return df.loc[keep].copy()


def default_cfg(base_cfg: dict, optimized_cfg: dict, compare_default: bool) -> dict | None:
    if not compare_default:
        return None
    cfg = base_cfg.get("default")
    if cfg is None or not cfg.get("path"):
        raise ValueError('plots.compare_default is true, so a top-level "default.path" is required.')
    return {"x": optimized_cfg["x"], "y": optimized_cfg["y"], "fields": optimized_cfg["fields"], **cfg}


def triangulated_field(ax, x, y, field, *, cmap, vmin=None, vmax=None):
    triang = tri.Triangulation(x, y)
    tris = triang.triangles
    edges = np.maximum.reduce([
        np.hypot(x[tris[:, 0]] - x[tris[:, 1]], y[tris[:, 0]] - y[tris[:, 1]]),
        np.hypot(x[tris[:, 1]] - x[tris[:, 2]], y[tris[:, 1]] - y[tris[:, 2]]),
        np.hypot(x[tris[:, 2]] - x[tris[:, 0]], y[tris[:, 2]] - y[tris[:, 0]]),
    ])
    triang.set_mask(edges > 0.15)
    levels = np.linspace(vmin, vmax, 51) if vmin is not None and vmax is not None else 50
    return ax.tricontourf(triang, field, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)


def save_field_plot(
    x,
    y,
    field_values,
    *,
    title: str,
    label: str,
    path: Path,
    cmap="viridis",
    vmin=None,
    vmax=None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    contour = triangulated_field(ax, x, y, field_values, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(label, fontsize=AXIS_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.set_xlabel("X Coordinate", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Y Coordinate", fontsize=AXIS_LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.axis("equal")
    ax.grid(True)
    fig.savefig(path)
    plt.close(fig)


def interpolate_to_target(ref: dict[str, np.ndarray], target: dict[str, np.ndarray], field: str) -> np.ndarray:
    points = np.column_stack((ref["x"], ref["y"]))
    target_points = np.column_stack((target["x"], target["y"]))
    interp = griddata(points, ref[field], target_points, method="linear")
    if np.any(~np.isfinite(interp)):
        nearest = griddata(points, ref[field], target_points, method="nearest")
        interp = np.where(np.isfinite(interp), interp, nearest)
    return interp


def safe_std(values_: np.ndarray) -> float:
    finite = values_[np.isfinite(values_)]
    if finite.size == 0:
        return 1.0
    return max(float(np.std(finite)), 1e-8)


def error_field(ref: dict[str, np.ndarray], opt: dict[str, np.ndarray], field: str, *, normalize: bool) -> np.ndarray:
    ref_on_opt = interpolate_to_target(ref, opt, field)
    error = opt[field] - ref_on_opt
    return error / safe_std(ref_on_opt) if normalize else error


def shared_field_limits(fields: list[str], datasets: list[dict[str, np.ndarray]]) -> dict[str, tuple[float, float]]:
    limits = {}
    for field in fields:
        finite_values = [
            np.asarray(data[field], dtype=float)[np.isfinite(data[field])]
            for data in datasets
            if data is not None and field in data
        ]
        combined = np.concatenate(finite_values) if finite_values else np.array([])
        limits[field] = (0.0, 1.0) if combined.size == 0 else (float(combined.min()), float(combined.max()))
    return limits


def error_limits(
    fields: list[str],
    comparisons: list[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]],
    *,
    normalize: bool,
) -> dict[str, float]:
    limits = {}
    for field in fields:
        maxima = []
        for ref, opt in comparisons:
            err = error_field(ref, opt, field, normalize=normalize)
            finite = np.abs(err[np.isfinite(err)])
            if finite.size:
                maxima.append(float(finite.max()))
        limit = max(maxima) if maxima else 1e-12
        limits[field] = limit if np.isfinite(limit) and limit > 0.0 else 1e-12
    return limits


def stitch(image_paths: list[Path], output_path: Path, *, layout: str) -> None:
    from PIL import Image, ImageOps

    images = [Image.open(path).convert("RGB") for path in image_paths]
    if layout == "horizontal":
        height = max(img.height for img in images)
        padded = [ImageOps.expand(img, border=(0, 0, 0, height - img.height), fill="white") for img in images]
        canvas = Image.new("RGB", (sum(img.width for img in padded), height), "white")
        offset = 0
        for img in padded:
            canvas.paste(img, (offset, 0))
            offset += img.width
    else:
        width = max(img.width for img in images)
        padded = [ImageOps.expand(img, border=(0, 0, width - img.width, 0), fill="white") for img in images]
        canvas = Image.new("RGB", (width, sum(img.height for img in padded)), "white")
        offset = 0
        for img in padded:
            canvas.paste(img, (0, offset))
            offset += img.height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def save_comparison(
    *,
    ref: dict[str, np.ndarray],
    opt: dict[str, np.ndarray],
    field: str,
    opt_label: str,
    output_path: Path,
    vmin: float,
    vmax: float,
    error_limit: float,
    normalize_error: bool,
    layout: str,
) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        paths = [
            tmp / f"01_DNS_{field}.png",
            tmp / f"02_{opt_label}_{field}.png",
            tmp / f"03_error_{field}.png",
        ]
        save_field_plot(
            ref["x"], ref["y"], ref[field],
            title=f"DNS {field}",
            label=field,
            path=paths[0],
            vmin=vmin,
            vmax=vmax,
        )
        save_field_plot(
            opt["x"], opt["y"], opt[field],
            title=f"{opt_label} {field}",
            label=field,
            path=paths[1],
            vmin=vmin,
            vmax=vmax,
        )
        label = f"{field} normalized difference" if normalize_error else f"{field} difference"
        save_field_plot(
            opt["x"], opt["y"], error_field(ref, opt, field, normalize=normalize_error),
            title=f"Error {opt_label} - DNS {field}",
            label=label,
            path=paths[2],
            cmap="coolwarm",
            vmin=-error_limit,
            vmax=error_limit,
        )
        stitch(paths, output_path, layout=layout)


def load_profile_frames(opt_cfg: dict, dns_cfg: dict, def_cfg: dict | None):
    opt = read_table(opt_cfg["path"])
    dns = read_table(dns_cfg["path"])
    default = read_table(def_cfg["path"]) if def_cfg is not None else None
    return opt, dns, default


def plot_profiles(
    *,
    x_locations: list[float],
    fields: list[str],
    opt: pd.DataFrame,
    dns: pd.DataFrame,
    opt_cfg: dict,
    dns_cfg: dict,
    output_dir: Path,
    tol: float,
    default: pd.DataFrame | None,
    default_cfg_: dict | None,
    case_name: str,
) -> None:
    missing_opt = [field for field in fields if field not in opt_cfg["fields"]]
    missing_dns = [field for field in fields if field not in dns_cfg["fields"]]
    if missing_opt:
        raise ValueError(f"Unknown optimized fields: {missing_opt}. Choose from {list(opt_cfg['fields'])}")
    if missing_dns:
        raise ValueError(f"Unknown DNS fields: {missing_dns}. Choose from {list(dns_cfg['fields'])}")

    for field in fields:
        fig, axes = plt.subplots(1, len(x_locations), figsize=(5 * len(x_locations), 4), squeeze=False)
        for idx, x_val in enumerate(x_locations):
            ax = axes[0][idx]
            opt_slice = opt[np.abs(column(opt, opt_cfg["x"]) - x_val) < tol].sort_values(
                by=opt_cfg["y"] if isinstance(opt_cfg["y"], str) else opt.columns[opt_cfg["y"]]
            )
            dns_slice = dns[np.abs(column(dns, dns_cfg["x"]) - x_val) < tol].sort_values(
                by=dns_cfg["y"] if isinstance(dns_cfg["y"], str) else dns.columns[dns_cfg["y"]]
            )
            ax.plot(column(opt_slice, opt_cfg["fields"][field]), column(opt_slice, opt_cfg["y"]),
                    color="tab:red", linewidth=1.5, alpha=0.7, label="Optimized")
            ax.plot(column(dns_slice, dns_cfg["fields"][field]), column(dns_slice, dns_cfg["y"]),
                    color="tab:blue", linewidth=1.5, alpha=0.7, label="DNS")

            if default is not None and default_cfg_ is not None:
                default_slice = default[
                    np.abs(column(default, default_cfg_["x"]) - x_val) < tol
                ].sort_values(
                    by=default_cfg_["y"] if isinstance(default_cfg_["y"], str) else default.columns[default_cfg_["y"]]
                )
                ax.plot(column(default_slice, default_cfg_["fields"][field]), column(default_slice, default_cfg_["y"]),
                        color="tab:green", linewidth=1.5, alpha=0.7, label="Default")

            ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
            ax.set_xlabel(field, fontsize=AXIS_LABEL_SIZE)
            ax.set_ylabel("y", fontsize=AXIS_LABEL_SIZE)
            ax.set_title(f"{field} — x = {x_val}  (tol = {tol})", fontsize=TITLE_SIZE)
            ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
            ax.legend(fontsize=LEGEND_SIZE)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        path = output_dir / f"profiles_{case_name}_{field}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
        print(f"Saved: {path}")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    cfg = load_config(args.config)
    plots = cfg["plots"]
    fields = plots["fields"]
    opt_cfg = cfg.get("optimized", cfg.get("simulation"))
    if opt_cfg is None:
        raise KeyError('JSON must define an "optimized" block.')

    output_dir = repo_path(cfg.get("output_dir", Path("results") / "experiments" / cfg["name"] / "plots"))
    opt_dir = output_dir / "optimized"
    default_dir = output_dir / "default"
    profiles_dir = output_dir / "profiles"
    for directory in (opt_dir, default_dir, profiles_dir):
        directory.mkdir(parents=True, exist_ok=True)

    compare_default = bool(plots.get("compare_default", False))
    def_cfg = default_cfg(cfg, opt_cfg, compare_default)
    dns = load_dataset(cfg["dns"])
    opt = load_dataset(opt_cfg)
    default = load_dataset(def_cfg) if def_cfg is not None else None

    if bool(plots.get("mask_hill", cfg.get("mask_hill", False))):
        dns = apply_hill_mask(dns)
        opt = apply_hill_mask(opt)
        if default is not None:
            default = apply_hill_mask(default)

    datasets = [dns, opt] + ([default] if default is not None else [])
    value_limits = shared_field_limits(fields, datasets)
    normalize_error = bool(plots.get("normalize_error", True))
    comparisons = [(dns, opt)] + ([(dns, default)] if default is not None else [])
    err_limits = error_limits(fields, comparisons, normalize=normalize_error)
    layout = plots.get("comparison_layout", "horizontal")

    for field in fields:
        lo, hi = value_limits[field]
        opt_path = opt_dir / f"Optimized_DNS_{cfg['name']}_{field}.png"
        save_comparison(
            ref=dns,
            opt=opt,
            field=field,
            opt_label="Optimized",
            output_path=opt_path,
            vmin=lo,
            vmax=hi,
            error_limit=err_limits[field],
            normalize_error=normalize_error,
            layout=layout,
        )
        print(f"Saved: {opt_path}")

        if default is not None:
            default_path = default_dir / f"Default_DNS_{cfg['name']}_{field}.png"
            save_comparison(
                ref=dns,
                opt=default,
                field=field,
                opt_label="Default",
                output_path=default_path,
                vmin=lo,
                vmax=hi,
                error_limit=err_limits[field],
                normalize_error=normalize_error,
                layout=layout,
            )
            print(f"Saved: {default_path}")

    make_profiles = bool(plots.get("make_profiles", bool(plots.get("x_locations", []))))
    if make_profiles:
        opt_frame, dns_frame, default_frame = load_profile_frames(opt_cfg, cfg["dns"], def_cfg)
        if bool(plots.get("mask_hill", cfg.get("mask_hill", False))):
            opt_frame = mask_frame(opt_frame, opt_cfg["x"], opt_cfg["y"])
            dns_frame = mask_frame(dns_frame, cfg["dns"]["x"], cfg["dns"]["y"])
            if default_frame is not None and def_cfg is not None:
                default_frame = mask_frame(default_frame, def_cfg["x"], def_cfg["y"])
        plot_profiles(
            x_locations=plots["x_locations"],
            fields=fields,
            opt=opt_frame,
            dns=dns_frame,
            opt_cfg=opt_cfg,
            dns_cfg=cfg["dns"],
            output_dir=profiles_dir,
            tol=plots["x_tol"],
            default=default_frame,
            default_cfg_=def_cfg,
            case_name=cfg["name"],
        )


if __name__ == "__main__":
    main()
