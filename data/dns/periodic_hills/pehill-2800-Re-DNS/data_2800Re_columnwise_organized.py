"""
Reorganise Balakumar periodic-hill dns-avg.dat from Tecplot block format
to a flat column-wise ASCII file.

Input:  dns-avg.dat  (same folder as this script)
Output: dns_avg_Re2800_columnwise_organized.ascii  (same folder)

Columns: nodenumber  x  y  u  v  density  production  dissipation  prod/diss  k  shear
"""

from pathlib import Path
import re
import numpy as np

HERE   = Path(__file__).resolve().parent
INPUT  = HERE / "dns-avg.dat"
OUTPUT = HERE / "dns_avg_Re2800_columnwise_organized.ascii"

VARS = ["X", "Y", "UAVG", "VAVG", "DAVG", "PROD", "DISSIP", "PRO/DISS", "KE", "SHEAR"]
HEADER = (
    "nodenumber       x-coordinate       y-coordinate       x-velocity       "
    "y-velocity         density            production       dissipation        "
    "prod-over-diss     k                shear\n"
)


def parse(path: Path) -> dict[str, list[str]]:
    """Read Tecplot BLOCK zones, keeping values as raw strings."""
    lines = path.read_text().splitlines()
    data: dict[str, list[str]] = {v: [] for v in VARS}

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("ZONE"):
            i += 1
            continue

        n_pts = int(re.search(r"I\s*=\s*(\d+)", line).group(1)) * \
                int(re.search(r"J\s*=\s*(\d+)", line).group(1))

        i += 1
        tokens: list[str] = []
        while i < len(lines) and len(tokens) < len(VARS) * n_pts:
            if lines[i].strip().startswith("ZONE"):
                break
            tokens.extend(lines[i].split())
            i += 1

        for j, var in enumerate(VARS):
            data[var].extend(tokens[j * n_pts : (j + 1) * n_pts])

    return data


def main() -> None:
    data = parse(INPUT)

    # Deduplicate on (x, y) — convert to float only for this
    coords = np.column_stack([[float(v) for v in data["X"]],
                               [float(v) for v in data["Y"]]])
    _, keep = np.unique(coords, axis=0, return_index=True)
    keep = np.sort(keep)

    with OUTPUT.open("w") as f:
        f.write(HEADER)
        for row, idx in enumerate(keep, start=1):
            f.write(f"{row:10d}  " + "  ".join(data[var][idx] for var in VARS) + "\n")

    print(f"Written: {OUTPUT}  ({len(keep)} points)")


if __name__ == "__main__":
    main()
