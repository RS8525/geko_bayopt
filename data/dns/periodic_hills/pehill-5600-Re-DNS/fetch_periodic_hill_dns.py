"""Fetch all 29 periodic-hill DNS cases and build a combined CSV per case.

Downloads the three ASCII files (mean_files.dat, rms_files1.dat, rms_files2.dat)
for every case in the para-database-for-PIML periodic-hill set
(https://github.com/xiaoh/para-database-for-PIML), and for each case writes

    data/dns/periodic_hill/<case>/periodic_hill_dns.csv

with columns x-coordinate, y-coordinate, x-velocity, y-velocity, TKE, pressure
(via build_periodic_hill_dns.build_dataset).

Case naming follows the database: ``alph<slope>-<...>-<Ly>`` e.g. ``alph10-9-3036``
is alpha=1.0, Ly/H=3.036. The slope parameter alpha is encoded in the prefix
(alph05=0.5, alph075=0.75, alph10=1.0, alph125=1.25, alph15=1.5).

By default the large raw .dat files are deleted after the CSV is built (each
case is ~120 MB raw; all 29 are ~3.5 GB). Pass --keep-raw to keep them.
The script is idempotent: existing per-case CSVs are skipped unless --force.

Usage:
    python data/dns/periodic_hills/pehill-5600-Re-DNS/fetch_periodic_hill_dns.py
    python data/dns/periodic_hills/pehill-5600-Re-DNS/fetch_periodic_hill_dns.py --out-dir data/dns/periodic_hills/pehill-5600-Re-DNS --keep-raw
    python data/dns/periodic_hills/pehill-5600-Re-DNS/fetch_periodic_hill_dns.py --cases alph10-9-3036 alph05-4071-2024
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Reuse the combiner that lives next to this script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_periodic_hill_dns import build_dataset, write_output  # noqa: E402

RAW_BASE = (
    "https://raw.githubusercontent.com/xiaoh/para-database-for-PIML/master/"
    "pehill-29-cases-DNS"
)
FILES = ("mean_files.dat", "rms_files1.dat", "rms_files2.dat")

# The 29 cases in the database.
CASES = [
    "alph05-10071-2024", "alph05-10071-3036", "alph05-10071-4048",
    "alph05-4071-2024", "alph05-4071-3036", "alph05-4071-4048",
    "alph05-7071-2024", "alph05-7071-3036", "alph05-7071-4048",
    "alph075-80355-3036",
    "alph10-12-2024", "alph10-12-3036", "alph10-12-4048",
    "alph10-6-2024", "alph10-6-3036", "alph10-6-4048",
    "alph10-9-2024", "alph10-9-3036", "alph10-9-4048",
    "alph125-99645-3036",
    "alph15-10929-2024", "alph15-10929-3036", "alph15-10929-4048",
    "alph15-13929-2024", "alph15-13929-3036", "alph15-13929-4048",
    "alph15-7929-2024", "alph15-7929-3036", "alph15-7929-4048",
]


def _download(url: str, dest: Path, retries: int = 3) -> None:
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, open(dest, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)  # 1 MB
                    if not chunk:
                        break
                    fh.write(chunk)
            return
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                raise RuntimeError(f"Failed to download {url}: {exc}") from exc
            print(f"    retry {attempt}/{retries} for {dest.name} ({exc})")


def process_case(case: str, out_dir: Path, keep_raw: bool, force: bool) -> str:
    case_dir = out_dir / case
    csv_path = case_dir / "periodic_hill_dns.csv"
    if csv_path.is_file() and not force:
        return "skipped (exists)"

    case_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for fname in FILES:
        dest = case_dir / fname
        if not dest.is_file():
            print(f"    downloading {fname} ...")
            _download(f"{RAW_BASE}/{case}/{fname}", dest)
        paths[fname] = dest

    data = build_dataset(paths["mean_files.dat"], paths["rms_files1.dat"], paths["rms_files2.dat"])
    write_output(data, csv_path, fmt="csv")

    if not keep_raw:
        for fname in FILES:
            paths[fname].unlink(missing_ok=True)

    return f"ok ({data.shape[0]} rows, TKE max={data[:,4].max():.4f})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("data/dns/periodic_hill"),
                    help="Base output folder (default: data/dns/periodic_hill)")
    ap.add_argument("--cases", nargs="*", default=None,
                    help="Subset of case names to fetch (default: all 29)")
    ap.add_argument("--keep-raw", action="store_true",
                    help="Keep the raw .dat files (default: delete after combining)")
    ap.add_argument("--force", action="store_true",
                    help="Rebuild even if the per-case CSV already exists")
    args = ap.parse_args()

    cases = args.cases or CASES
    unknown = [c for c in cases if c not in CASES]
    if unknown:
        print(f"WARNING: not in the known 29 cases (will still try): {unknown}")

    print(f"Fetching {len(cases)} case(s) into {args.out_dir}")
    failures = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case}")
        try:
            status = process_case(case, args.out_dir, args.keep_raw, args.force)
            print(f"    -> {status}")
        except Exception as exc:  # noqa: BLE001
            print(f"    -> FAILED: {exc}")
            failures.append(case)

    print()
    print(f"Done. {len(cases) - len(failures)}/{len(cases)} cases built.")
    if failures:
        print(f"Failed: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
