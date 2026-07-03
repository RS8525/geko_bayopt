"""Combine Xiao/Laizet periodic-hill DNS files into one reference file.

Reads the three ASCII files from the para-database-for-PIML format
(https://github.com/xiaoh/para-database-for-PIML), all on the same grid:

    mean_files.dat : x, y, u, v, w, p            (mean fields)
    rms_files1.dat : x, y, <u'u'>, <v'v'>, <w'w'>, <p'p'>   (normal stresses)
    rms_files2.dat : x, y, <u'v'>, <u'w'>, <v'w'>           (shear stresses)

and writes a single file with columns:

    x-coordinate, y-coordinate, x-velocity, y-velocity, TKE, pressure

where TKE is the turbulent kinetic energy

    k = 0.5 * (<u'u'> + <v'v'> + <w'w'>)

NOTE on the rms files: despite the name, columns 3-5 of rms_files1 are the
velocity *variances* (mean-squares) <u'u'>, <v'v'>, <w'w'>, NOT their square
roots. This is verified by Cauchy-Schwarz: read as variances, |<u'v'>| <=
sqrt(<u'u'><v'v'>) holds everywhere; read as RMS it is violated almost
everywhere. So TKE uses the columns directly (no squaring).

All quantities are left in the database's native (non-dimensional) units
(U_bulk, H); the periodic-hills loader / extract apply any reference scaling.

Usage:
    python data/dns/periodic_hills/pehill-5600-Re-DNS/build_periodic_hill_dns.py
    python data/dns/periodic_hills/pehill-5600-Re-DNS/build_periodic_hill_dns.py --input-dir periodic_hills/pehill-5600-Re-DNS \
        --out periodic_hills/pehill-5600-Re-DNS/periodic_hill_dns.csv
    python data/dns/periodic_hills/pehill-5600-Re-DNS/build_periodic_hill_dns.py --format ascii
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Output column order requested for the cost-function target.
OUTPUT_COLUMNS = [
    "x-coordinate",
    "y-coordinate",
    "x-velocity",
    "y-velocity",
    "TKE",
    "pressure",
]


def _load(path: Path, expected_cols: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"DNS file not found: {path}")
    data = np.genfromtxt(path, dtype=float)
    if data.ndim != 2 or data.shape[1] != expected_cols:
        raise ValueError(
            f"{path.name}: expected {expected_cols} columns, got shape {data.shape}"
        )
    return data


def build_dataset(
    mean_path: Path, rms1_path: Path, rms2_path: Path | None = None
) -> np.ndarray:
    """Return an (N, 6) array with the OUTPUT_COLUMNS.

    ``rms2_path`` (shear stresses) is not needed to build TKE, but if given it
    is used only to validate that all files share the same grid.
    """
    mean = _load(mean_path, 6)   # x, y, u, v, w, p
    rms1 = _load(rms1_path, 6)   # x, y, <u'u'>, <v'v'>, <w'w'>, <p'p'>

    if mean.shape[0] != rms1.shape[0]:
        raise ValueError(
            f"Row count mismatch: {mean_path.name} has {mean.shape[0]} rows, "
            f"{rms1_path.name} has {rms1.shape[0]}."
        )
    if not np.allclose(mean[:, :2], rms1[:, :2], atol=1e-8):
        raise ValueError(
            f"{mean_path.name} and {rms1_path.name} are not on the same (x, y) grid."
        )
    if rms2_path is not None:
        rms2 = _load(rms2_path, 5)  # x, y, <u'v'>, <u'w'>, <v'w'>
        if rms2.shape[0] != mean.shape[0] or not np.allclose(
            mean[:, :2], rms2[:, :2], atol=1e-8
        ):
            raise ValueError(
                f"{rms2_path.name} is not on the same grid as {mean_path.name}."
            )

    x = mean[:, 0]
    y = mean[:, 1]
    u = mean[:, 2]
    v = mean[:, 3]
    p = mean[:, 5]
    tke = 0.5 * (rms1[:, 2] + rms1[:, 3] + rms1[:, 4])

    if np.any(tke < 0):
        n_neg = int(np.sum(tke < 0))
        raise ValueError(
            f"Computed TKE is negative at {n_neg} points -- check that "
            f"{rms1_path.name} holds variances (not RMS) in columns 3-5."
        )

    return np.column_stack([x, y, u, v, tke, p])


def write_output(data: np.ndarray, out_path: Path, fmt: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join(OUTPUT_COLUMNS) if fmt == "csv" else " ".join(OUTPUT_COLUMNS)
    delimiter = "," if fmt == "csv" else " "
    comments = "" if fmt == "csv" else "# "
    np.savetxt(
        out_path,
        data,
        delimiter=delimiter,
        header=header,
        comments=comments,
        fmt="%.10g",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=Path("data/dns/periodic_hills/pehill-5600-Re-DNS"),
                    help="Folder containing the three .dat files (default: data/dns/periodic_hill)")
    ap.add_argument("--mean", default="mean_files.dat", help="Mean fields filename")
    ap.add_argument("--rms1", default="rms_files1.dat", help="Normal-stress filename")
    ap.add_argument("--rms2", default="rms_files2.dat", help="Shear-stress filename (grid check only)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output file (default: <input-dir>/periodic_hill_dns.csv)")
    ap.add_argument("--format", choices=["csv", "ascii"], default="csv",
                    help="Output format (default: csv)")
    args = ap.parse_args()

    mean_path = args.input_dir / args.mean
    rms1_path = args.input_dir / args.rms1
    rms2_path = args.input_dir / args.rms2
    rms2_arg = rms2_path if rms2_path.is_file() else None

    out_path = args.out
    if out_path is None:
        ext = "csv" if args.format == "csv" else "ascii"
        out_path = args.input_dir / f"periodic_hill_dns.{ext}"

    print(f"Reading from {args.input_dir} ...")
    data = build_dataset(mean_path, rms1_path, rms2_arg)

    print("Field ranges in the combined dataset:")
    for i, name in enumerate(OUTPUT_COLUMNS):
        col = data[:, i]
        print(f"  {name:12s}: min={col.min():+.5g}  max={col.max():+.5g}  mean={col.mean():+.5g}")

    write_output(data, out_path, args.format)
    print(f"Wrote {data.shape[0]} rows x {data.shape[1]} cols -> {out_path}")


if __name__ == "__main__":
    main()
