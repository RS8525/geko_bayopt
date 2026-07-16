"""Plot the FFS common evaluation grid or exported node cloud.

The default output is a report-friendly diagram of the Cartesian common grid
used for the final FFS field-error evaluation. A secondary node-cloud mode is
kept for debugging Fluent exports.

By default the script also recomputes and prints the Re=3000 common-grid
sensitivity table used in the FFS appendix.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


DEFAULT_DNS = Path("data/dns/ffs/FFS_Reh2000_SBES_Node_2D.csv")
DEFAULT_MESH = Path("results/fluent/ffs_re2000_all_param_mae_final/FFS_3_2d_v2.msh.h5")
DEFAULT_ASCII = Path(
    "results/ffs_final_runs/all_param_runs/ffs_re2000_all_param_mae_final/"
    "ffs_Reh2000_geko_default.ascii"
)
DEFAULT_OUTPUT = Path("docs/figures/ffs_plots/appendix/ffs_common_evaluation_grid.png")
DEFAULT_SENSITIVITY_CONFIG = Path("configs/ffs_final/all_param/ffs_re3000_all_param.json")
DEFAULT_SENSITIVITY_DEFAULT_ASCII = Path(
    "results/ffs_final_runs/all_param_runs/ffs_re3000_all_param_final/"
    "ffs_Reh3000_geko_default.ascii"
)
DEFAULT_SENSITIVITY_OPTIMIZED_ASCII = Path(
    "results/ffs_final_runs/all_param_runs/ffs_re3000_all_param_final/"
    "ffs_Reh3000_Csep1.4807_Cnw0.5081_Cmix0.9028_Cjet0.977_Cturb1.5.ascii"
)
DEFAULT_SENSITIVITY_OUTPUT = Path(
    "docs/ffs_results_section/tables/selected/appendix/"
    "common_grid_sensitivity_re3000.csv"
)
DEFAULT_SENSITIVITY_RESOLUTIONS = (
    (180, 60),
    (270, 90),
    (360, 120),
    (540, 180),
    (720, 240),
)


def ffs_floor(x: np.ndarray | float) -> np.ndarray | float:
    """Piecewise floor used by the FFS common-grid mask."""

    return np.where(np.asarray(x) < 0.0, -0.01, 0.0)


def load_dns_bounds(path: Path) -> tuple[float, float, float, float]:
    frame = pd.read_csv(path, usecols=["x-coordinate", "y-coordinate"])
    return (
        float(frame["x-coordinate"].min()),
        float(frame["x-coordinate"].max()),
        float(frame["y-coordinate"].min()),
        float(frame["y-coordinate"].max()),
    )


def data_to_pixels(
    x: np.ndarray | float,
    y: np.ndarray | float,
    ranges: tuple[float, float, float, float],
    width: int,
    height: int,
    margin: int,
) -> tuple[np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = ranges
    dx = xmax - xmin
    dy = ymax - ymin
    if dx <= 0 or dy <= 0:
        raise ValueError("Degenerate coordinate range")

    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    scale = min(plot_w / dx, plot_h / dy)
    used_w = dx * scale
    used_h = dy * scale
    xoff = margin + 0.5 * (plot_w - used_w)
    yoff = margin + 0.5 * (plot_h - used_h)

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    px = xoff + (x_arr - xmin) * scale
    py = height - (yoff + (y_arr - ymin) * scale)
    return px, py


def draw_data_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    ranges: tuple[float, float, float, float],
    width: int,
    height: int,
    margin: int,
    *,
    fill: tuple[int, int, int],
    line_width: int = 1,
) -> None:
    x, y = data_to_pixels(
        np.array([start[0], end[0]]),
        np.array([start[1], end[1]]),
        ranges,
        width,
        height,
        margin,
    )
    draw.line((float(x[0]), float(y[0]), float(x[1]), float(y[1])), fill=fill, width=line_width)


def draw_data_polygon(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    ranges: tuple[float, float, float, float],
    width: int,
    height: int,
    margin: int,
    *,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int] | None = None,
) -> None:
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    px, py = data_to_pixels(xs, ys, ranges, width, height, margin)
    draw.polygon([(float(x), float(y)) for x, y in zip(px, py)], fill=fill, outline=outline)


def render_common_grid(
    dns_path: Path,
    output: Path,
    *,
    nx: int,
    ny: int,
    grid_stride_x: int,
    grid_stride_y: int,
) -> Path:
    width, height = 2400, 600
    margin = 40
    output.parent.mkdir(parents=True, exist_ok=True)

    ranges = load_dns_bounds(dns_path)
    xmin, xmax, ymin, ymax = ranges
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(x, y)
    keep = yy.ravel() >= (ffs_floor(xx.ravel()) - 1e-12)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    # Solid part downstream of the step; this is excluded from the objective.
    draw_data_polygon(
        draw,
        [(0.0, ymin), (xmax, ymin), (xmax, 0.0), (0.0, 0.0)],
        ranges,
        width,
        height,
        margin,
        fill=(228, 228, 228),
        outline=(150, 150, 150),
    )

    grid_color = (190, 202, 214)
    for xi in x[:: max(1, grid_stride_x)]:
        y0 = float(ffs_floor(float(xi)))
        draw_data_line(
            draw,
            (float(xi), y0),
            (float(xi), ymax),
            ranges,
            width,
            height,
            margin,
            fill=grid_color,
        )

    for yi in y[:: max(1, grid_stride_y)]:
        if yi < 0.0:
            start = (xmin, float(yi))
            end = (0.0, float(yi))
        else:
            start = (xmin, float(yi))
            end = (xmax, float(yi))
        draw_data_line(
            draw,
            start,
            end,
            ranges,
            width,
            height,
            margin,
            fill=grid_color,
        )

    boundary_color = (20, 20, 20)
    draw_data_line(draw, (xmin, ymin), (0.0, ymin), ranges, width, height, margin, fill=boundary_color, line_width=3)
    draw_data_line(draw, (0.0, ymin), (0.0, 0.0), ranges, width, height, margin, fill=boundary_color, line_width=3)
    draw_data_line(draw, (0.0, 0.0), (xmax, 0.0), ranges, width, height, margin, fill=boundary_color, line_width=3)
    draw_data_line(draw, (xmin, ymax), (xmax, ymax), ranges, width, height, margin, fill=boundary_color, line_width=2)
    draw_data_line(draw, (xmin, ymin), (xmin, ymax), ranges, width, height, margin, fill=boundary_color, line_width=2)
    draw_data_line(draw, (xmax, 0.0), (xmax, ymax), ranges, width, height, margin, fill=boundary_color, line_width=2)

    image.save(output)

    metadata = output.with_suffix(".txt")
    metadata.write_text(
        "\n".join(
            [
                f"source=dns:{dns_path.as_posix()}",
                f"output={output.as_posix()}",
                f"common_grid_nx={nx}",
                f"common_grid_ny={ny}",
                f"total_grid_points={nx * ny}",
                f"evaluated_points={int(keep.sum())}",
                f"masked_solid_step_points={int((~keep).sum())}",
                f"x_min={xmin:.12g}",
                f"x_max={xmax:.12g}",
                f"y_min={ymin:.12g}",
                f"y_max={ymax:.12g}",
                f"displayed_x_stride={max(1, grid_stride_x)}",
                f"displayed_y_stride={max(1, grid_stride_y)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata


def _try_load_hdf5_points(path: Path):
    try:
        import h5py  # type: ignore
    except ImportError:
        return None, "h5py not installed"

    candidates = []

    def visit(name, obj):
        if not hasattr(obj, "shape") or not hasattr(obj, "dtype"):
            return
        shape = tuple(obj.shape)
        if len(shape) != 2 or shape[0] < 100 or shape[1] not in (2, 3):
            return
        lname = name.lower()
        score = sum(1 for key in ("coord", "node", "vertex", "point") if key in lname)
        candidates.append((score, name, obj))

    with h5py.File(path, "r") as h5:
        h5.visititems(visit)
        if not candidates:
            return None, "no coordinate-like dataset found"
        candidates.sort(key=lambda item: (item[0], item[2].shape[0]), reverse=True)
        _, name, dataset = candidates[0]
        arr = dataset[()]

    arr = np.asarray(arr, dtype=float)
    if arr.shape[1] == 3:
        arr = arr[:, :2]
    return arr, f"hdf5:{name}"


def load_node_points(mesh_path: Path, ascii_path: Path | None):
    if mesh_path.is_file():
        points, source = _try_load_hdf5_points(mesh_path)
        if points is not None:
            return points[:, 0], points[:, 1], source

    if ascii_path is None:
        raise RuntimeError(
            "Could not read the HDF5 mesh. Install h5py or pass --ascii PATH "
            "to plot exported Fluent node coordinates."
        )

    frame = pd.read_csv(ascii_path, sep=r"\s+", engine="python")
    frame.columns = frame.columns.str.strip()
    return (
        frame["x-coordinate"].astype(float).to_numpy(),
        frame["y-coordinate"].astype(float).to_numpy(),
        f"ascii:{ascii_path.as_posix()}",
    )


def render_node_cloud(x, y, output: Path, *, max_points: int, source: str) -> Path:
    width, height = 2400, 600
    margin = 40
    output.parent.mkdir(parents=True, exist_ok=True)

    n_total = len(x)
    if n_total > max_points:
        step = max(1, math.ceil(n_total / max_points))
        x_plot = x[::step]
        y_plot = y[::step]
    else:
        step = 1
        x_plot = x
        y_plot = y

    ranges = (float(x.min()), float(x.max()), float(y.min()), float(y.max()))
    px, py = data_to_pixels(x_plot, y_plot, ranges, width, height, margin)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    for xi, yi in zip(px, py):
        draw.point((float(xi), float(yi)), fill=(35, 35, 35))

    draw.rectangle((margin, margin, width - margin, height - margin), outline=(0, 0, 0), width=2)
    image.save(output)

    metadata = output.with_suffix(".txt")
    xmin, xmax, ymin, ymax = ranges
    metadata.write_text(
        "\n".join(
            [
                f"source={source}",
                f"output={output.as_posix()}",
                f"total_points={n_total}",
                f"plotted_points={len(x_plot)}",
                f"sampling_step={step}",
                f"x_min={xmin:.12g}",
                f"x_max={xmax:.12g}",
                f"y_min={ymin:.12g}",
                f"y_max={ymax:.12g}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata


def parse_resolutions(value: str) -> list[tuple[int, int]]:
    resolutions = []
    for raw_part in value.split(","):
        part = raw_part.strip().lower()
        if not part:
            continue
        if "x" not in part:
            raise ValueError(
                "Sensitivity resolutions must be formatted like "
                "'180x60,360x120'."
            )
        nx_raw, ny_raw = part.split("x", 1)
        nx = int(nx_raw)
        ny = int(ny_raw)
        if nx <= 0 or ny <= 0:
            raise ValueError("Sensitivity grid dimensions must be positive.")
        resolutions.append((nx, ny))
    if not resolutions:
        raise ValueError("At least one sensitivity resolution is required.")
    return resolutions


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src = repo_root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def compute_common_grid_sensitivity(
    *,
    config_path: Path,
    default_ascii: Path,
    optimized_ascii: Path,
    output_csv: Path,
    resolutions: list[tuple[int, int]],
) -> list[dict[str, float | int | str]]:
    """Recompute the Re=3000 common-grid sensitivity table.

    This intentionally uses the package's own FFS loader and
    ``FieldErrorCalculator`` so the appendix table follows the same objective
    path as the optimization results.
    """

    _ensure_src_on_path()
    try:
        from geko_bayesopt.cases.ffs.case import ForwardFacingStepCase
        from geko_bayesopt.fluent.mesh_config import MeshConfig
        from geko_bayesopt.objective.field_error import FieldErrorCalculator
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Common-grid sensitivity requires the project Python environment "
            "with SciPy installed. Run for example: "
            ".\\.venv\\Scripts\\python.exe scripts\\ffs\\plot_ffs_mesh.py"
        ) from exc

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    mesh_config = MeshConfig(**config["mesh"])
    flow_case = ForwardFacingStepCase(config["case"]["options"], mesh_config)
    dns_coords, dns_fields = flow_case.load_dns(config["case"]["options"]["dns_path"])
    fields = ["Ux", "total-turbulent-kinetic-energy"]

    runs = [
        (
            "default",
            flow_case.build_run_result(
                run_id="default",
                parameters={},
                ascii_path=default_ascii,
            ),
        ),
        (
            "optimized",
            flow_case.build_run_result(
                run_id="optimized",
                parameters={},
                ascii_path=optimized_ascii,
            ),
        ),
    ]

    rows: list[dict[str, float | int | str]] = []
    totals: dict[tuple[int, int, str], float] = {}
    for nx, ny in resolutions:
        calculator = FieldErrorCalculator(
            dns_coords,
            {field: dns_fields[field] for field in fields},
            mask_hill=False,
            evaluation_mode="common_grid",
            common_grid_nx=nx,
            common_grid_ny=ny,
            common_grid_floor="ffs_step",
            field_error_norm="l2",
        )
        for run_name, run in runs:
            ux_error = calculator.calculate_error(
                run.grid_coords,
                run.fields,
                field_name="Ux",
            )
            k_error = calculator.calculate_error(
                run.grid_coords,
                run.fields,
                field_name="total-turbulent-kinetic-energy",
            )
            total = ux_error + k_error
            row: dict[str, float | int | str] = {
                "grid": f"{nx}x{ny}",
                "nx": nx,
                "ny": ny,
                "run": run_name,
                "Ux": ux_error,
                "k_total": k_error,
                "total": total,
            }
            rows.append(row)
            totals[(nx, ny, run_name)] = total

    if (360, 120) not in [(nx, ny) for nx, ny in resolutions]:
        raise ValueError("Sensitivity resolutions must include 360x120 as reference.")

    for row in rows:
        reference = totals[(360, 120, str(row["run"]))]
        row["rel_to_360x120_percent"] = (float(row["total"]) / reference - 1.0) * 100.0

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "grid",
                "nx",
                "ny",
                "run",
                "Ux",
                "k_total",
                "total",
                "rel_to_360x120_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


def print_common_grid_sensitivity(rows: list[dict[str, float | int | str]]) -> None:
    print()
    print("Common-grid sensitivity, Re=3000, L2 field error")
    print("Relative change is measured against the selected 360x120 grid.")
    print(
        f"{'Grid':>10}  {'Run':>9}  {'Ux':>10}  {'k_total':>10}  "
        f"{'Total':>10}  {'Rel.':>9}"
    )
    for row in rows:
        print(
            f"{str(row['grid']):>10}  "
            f"{str(row['run']):>9}  "
            f"{float(row['Ux']):10.6f}  "
            f"{float(row['k_total']):10.6f}  "
            f"{float(row['total']):10.6f}  "
            f"{float(row['rel_to_360x120_percent']):+8.3f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("common-grid", "node-cloud"), default="common-grid")
    parser.add_argument("--dns", type=Path, default=DEFAULT_DNS)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--ascii", type=Path, default=DEFAULT_ASCII)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nx", type=int, default=360)
    parser.add_argument("--ny", type=int, default=120)
    parser.add_argument("--grid-stride-x", type=int, default=1)
    parser.add_argument("--grid-stride-y", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=120_000)
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Do not recompute and print the appendix common-grid sensitivity table.",
    )
    parser.add_argument(
        "--sensitivity-config",
        type=Path,
        default=DEFAULT_SENSITIVITY_CONFIG,
    )
    parser.add_argument(
        "--sensitivity-default-ascii",
        type=Path,
        default=DEFAULT_SENSITIVITY_DEFAULT_ASCII,
    )
    parser.add_argument(
        "--sensitivity-optimized-ascii",
        type=Path,
        default=DEFAULT_SENSITIVITY_OPTIMIZED_ASCII,
    )
    parser.add_argument(
        "--sensitivity-output",
        type=Path,
        default=DEFAULT_SENSITIVITY_OUTPUT,
    )
    parser.add_argument(
        "--sensitivity-resolutions",
        default=",".join(f"{nx}x{ny}" for nx, ny in DEFAULT_SENSITIVITY_RESOLUTIONS),
        help="Comma-separated grid list, for example '180x60,360x120,720x240'.",
    )
    args = parser.parse_args()

    if args.mode == "common-grid":
        metadata = render_common_grid(
            args.dns,
            args.output,
            nx=args.nx,
            ny=args.ny,
            grid_stride_x=args.grid_stride_x,
            grid_stride_y=args.grid_stride_y,
        )
    else:
        output = args.output
        if output == DEFAULT_OUTPUT:
            output = Path("docs/figures/ffs_plots/appendix/ffs_re2000_mesh_points.png")
        x, y, source = load_node_points(args.mesh, args.ascii)
        metadata = render_node_cloud(x, y, output, max_points=args.max_points, source=source)

    print(f"Wrote {output if args.mode == 'node-cloud' else args.output}")
    print(f"Wrote {metadata}")

    if not args.skip_sensitivity:
        try:
            rows = compute_common_grid_sensitivity(
                config_path=args.sensitivity_config,
                default_ascii=args.sensitivity_default_ascii,
                optimized_ascii=args.sensitivity_optimized_ascii,
                output_csv=args.sensitivity_output,
                resolutions=parse_resolutions(args.sensitivity_resolutions),
            )
        except RuntimeError as exc:
            print()
            print(f"Skipped common-grid sensitivity calculation: {exc}")
        else:
            print(f"Wrote {args.sensitivity_output}")
            print_common_grid_sensitivity(rows)


if __name__ == "__main__":
    main()
