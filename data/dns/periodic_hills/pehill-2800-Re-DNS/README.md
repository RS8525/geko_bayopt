# Periodic Hills DNS Data — Re=2800 (Balakumar)

Source: [NASA Turbulence Modeling Resource](https://tmbwg.github.io/turbmodels/Other_DNS_Data/2dhill_periodic_compress.html), DNS data provided by P. Balakumar.

Reference publication:
> Balakumar, P., "DNS/LES Simulations of Separated Flows at High Reynolds Numbers,"
> AIAA Paper 2015-2783, June 2015. https://doi.org/10.2514/6.2015-2783

---

## Case description

2-D separating flow over periodic hills, computed with a compressible DNS code
using a compact high-order scheme.

| Parameter | Value |
| Reynolds number | 2800 |
| Mach number | 0.2 |
| Streamwise hill spacing | L_x = 9h |
| Channel height | L_y = 3.035h |
| Spanwise extent | 4.5h |
| Grid resolution (x × y × z) | 801 × 351 × 513 |
| Separation point | x ≈ 0.233h |
| Reattachment point | x ≈ 5.50h |

U_b is the bulk velocity at the crest of the first hill. The flow is periodic
in the streamwise direction.

---

## Active file

```
dns_avg_Re2800_columnwise_organized.ascii
```

This file is derived from the original `dns-avg.dat.zip`.
The raw file is organised in
blocks — one block per x-station. The script
`data_2800Re_columnwise_organized.py` reorganises it into a flat
column-wise layout with one row per point, which is required in pipeline.

To regenerate the active file from the raw source:

```bash
python data/dns/periodic_hills/pehill-2800-Re-DNS/data_2800Re_columnwise_organized.py
```

---

## Column layout

The reorganised file has a one-line header followed by space-separated columns:

| Index | Field |
| 0 | node number |
| 1 | x-coordinate (normalised by h) |
| 2 | y-coordinate (normalised by h) |
| 3 | x-velocity |
| 4 | y-velocity |
| 5 | density |
| 6 | production |
| 7 | dissipation |
| 8 | production over dissipation |
| 9 | k (turbulent kinetic energy) |
| 10 | shear |


Refer to the original NASA TMR page and reference publicaion for 
complete variable list and non-dimensionalisation details.

