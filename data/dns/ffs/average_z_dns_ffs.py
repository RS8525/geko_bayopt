"""Reduce one or more 3D FFS DNS exports to spanwise-averaged 2D CSV files.

The exported mesh contains small coordinate noise and sparse intermediate
z-levels. Exact ``(x, y)`` grouping therefore does not reconstruct the swept
2D mesh. This script identifies the primary spanwise planes, maps every plane
to the canonical first-plane mesh, and performs a trapezoidal spanwise average.

Examples
--------
Convert every raw FFS export that does not yet have a 2D output:

    python data/dns/ffs/average_z_dns_ffs.py

Convert selected cases and replace existing outputs:

    python data/dns/ffs/average_z_dns_ffs.py \
        data/dns/ffs/FFS_Reh2000_SBES_Node \
        data/dns/ffs/FFS_Reh3000_SBES_Node \
        --overwrite
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


DEFAULT_DATA_DIR = Path(__file__).resolve().parent
RAW_PATTERN = "FFS_Reh*_SBES_Node"
COORD_COLUMNS = ["x-coordinate", "y-coordinate", "z-coordinate"]
NON_FIELD_COLUMNS = {"nodenumber", *COORD_COLUMNS}


def _read_header(path: Path) -> list[str]:
    columns = pd.read_csv(path, nrows=0).columns.str.strip().tolist()
    required = set(COORD_COLUMNS)
    missing = sorted(required.difference(columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return columns


def _read_coordinates(path: Path) -> pd.DataFrame:
    coords = pd.read_csv(
        path,
        usecols=lambda column: column.strip() in set(COORD_COLUMNS),
    )
    coords.columns = coords.columns.str.strip()
    return coords[COORD_COLUMNS]


def _build_plane_mapping(
    coords: pd.DataFrame,
    *,
    z_plane_count: int,
    plane_tolerance_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xyz = coords.to_numpy(dtype=float, copy=False)
    xy = xyz[:, :2]
    z = xyz[:, 2]

    z_min = float(np.min(z))
    z_max = float(np.max(z))
    if not z_max > z_min:
        raise ValueError("The input must contain more than one z-coordinate.")

    planes = np.linspace(z_min, z_max, z_plane_count)
    spacing = float(planes[1] - planes[0])
    nearest_plane = np.rint((z - z_min) / spacing).astype(np.int32)
    nearest_plane = np.clip(nearest_plane, 0, z_plane_count - 1)
    plane_distance = np.abs(z - planes[nearest_plane])
    primary_mask = plane_distance <= spacing * plane_tolerance_fraction

    reference_mask = primary_mask & (nearest_plane == 0)
    reference_xy = xy[reference_mask]
    reference_xy = np.unique(reference_xy, axis=0)
    if len(reference_xy) == 0:
        raise ValueError("Could not identify the canonical first z-plane.")

    tree = cKDTree(reference_xy)
    mapped_point = np.full(len(coords), -1, dtype=np.int32)
    distances, indices = tree.query(xy[primary_mask], workers=-1)
    mapped_point[primary_mask] = indices.astype(np.int32)

    primary_rows_per_plane = np.bincount(
        nearest_plane[primary_mask],
        minlength=z_plane_count,
    )
    expected = len(reference_xy)
    if np.any(primary_rows_per_plane < 0.98 * expected):
        raise ValueError(
            "A primary z-plane contains substantially fewer points than the "
            f"canonical plane ({primary_rows_per_plane.tolist()} vs {expected}). "
            "Check --z-planes and --plane-tolerance-fraction."
        )

    max_mapping_distance = float(np.max(distances))
    print(
        f"  canonical points: {expected:,}; primary planes: {z_plane_count}; "
        f"ignored sparse rows: {np.count_nonzero(~primary_mask):,}; "
        f"max xy mapping distance: {max_mapping_distance:.3g}"
    )

    group_id = np.full(len(coords), -1, dtype=np.int64)
    group_id[primary_mask] = (
        mapped_point[primary_mask].astype(np.int64) * z_plane_count
        + nearest_plane[primary_mask]
    )
    group_counts = np.bincount(
        group_id[primary_mask],
        minlength=len(reference_xy) * z_plane_count,
    )

    trapezoid_weights = np.ones(z_plane_count, dtype=float)
    trapezoid_weights[[0, -1]] = 0.5
    return reference_xy, nearest_plane, group_id, group_counts, trapezoid_weights


def average_file(
    input_path: Path,
    output_path: Path,
    *,
    z_plane_count: int,
    plane_tolerance_fraction: float,
    chunksize: int,
) -> None:
    start = time.time()
    columns = _read_header(input_path)
    field_columns = [column for column in columns if column not in NON_FIELD_COLUMNS]

    print(f"Reading coordinates from {input_path} ...")
    coords = _read_coordinates(input_path)
    (
        reference_xy,
        nearest_plane,
        group_id,
        group_counts,
        trapezoid_weights,
    ) = _build_plane_mapping(
        coords,
        z_plane_count=z_plane_count,
        plane_tolerance_fraction=plane_tolerance_fraction,
    )
    del coords

    field_sums = np.zeros((len(reference_xy), len(field_columns)), dtype=float)
    weight_sums = np.zeros(len(reference_xy), dtype=float)

    print(f"  averaging {len(field_columns)} numeric fields in chunks of {chunksize:,} rows")
    row_offset = 0
    for chunk in pd.read_csv(input_path, chunksize=chunksize):
        chunk.columns = chunk.columns.str.strip()
        row_count = len(chunk)
        row_slice = slice(row_offset, row_offset + row_count)
        chunk_group_id = group_id[row_slice]
        valid = chunk_group_id >= 0

        valid_group_id = chunk_group_id[valid]
        point_index = valid_group_id // z_plane_count
        plane_index = nearest_plane[row_slice][valid]
        duplicate_count = group_counts[valid_group_id]
        row_weights = trapezoid_weights[plane_index] / duplicate_count

        values = chunk.loc[valid, field_columns].to_numpy(dtype=float)
        np.add.at(field_sums, point_index, values * row_weights[:, None])
        np.add.at(weight_sums, point_index, row_weights)
        row_offset += row_count

    if row_offset != len(group_id):
        raise RuntimeError(f"Row count changed while reading {input_path}.")
    if np.any(weight_sums == 0.0):
        raise ValueError("At least one canonical point has no spanwise samples.")

    averaged = field_sums / weight_sums[:, None]
    output = pd.DataFrame(reference_xy, columns=["x-coordinate", "y-coordinate"])
    for column_index, column in enumerate(field_columns):
        output[column] = averaged[:, column_index]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, float_format="%.9g")
    print(
        f"  wrote {len(output):,} rows to {output_path} "
        f"in {time.time() - start:.1f} seconds"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help=f"Raw DNS exports. Default: {DEFAULT_DATA_DIR / RAW_PATTERN}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing *_2D.csv output.",
    )
    parser.add_argument(
        "--z-planes",
        type=int,
        default=20,
        help="Number of primary spanwise planes (default: 20).",
    )
    parser.add_argument(
        "--plane-tolerance-fraction",
        type=float,
        default=0.1,
        help="Maximum distance from a primary plane as a fraction of plane spacing.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="CSV rows processed per averaging chunk.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    inputs = args.inputs or sorted(DEFAULT_DATA_DIR.glob(RAW_PATTERN))
    if not inputs:
        raise FileNotFoundError(f"No inputs match {DEFAULT_DATA_DIR / RAW_PATTERN}")

    for input_path in inputs:
        input_path = input_path.resolve()
        if not input_path.is_file():
            raise FileNotFoundError(input_path)

        output_path = input_path.with_name(f"{input_path.name}_2D.csv")
        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing output: {output_path}")
            continue

        average_file(
            input_path,
            output_path,
            z_plane_count=args.z_planes,
            plane_tolerance_fraction=args.plane_tolerance_fraction,
            chunksize=args.chunksize,
        )


if __name__ == "__main__":
    main()
