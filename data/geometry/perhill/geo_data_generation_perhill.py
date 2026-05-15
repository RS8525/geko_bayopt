from geko_bayesopt.ansys.periodic_hill.geometry_generation.periodic_hill_geom import create_perhill_geom_ansys

# List of cases with alpha, L, and ceiling_param values
cases = [
    # (0.5, 10.071, 2.024),
    # (0.5, 10.071, 3.036),
    # (0.5, 10.071, 4.048),
    # (0.5, 4.071, 2.024),
    # (0.5, 4.071, 3.036),
    # (0.5, 4.071, 4.048),
    # (0.5, 7.071, 2.024),
    # (0.5, 7.071, 3.036),
    # (0.5, 7.071, 4.048),
     (0.75, 8.0355, 3.036),
    (1.0, 12, 2.024),
    (1.0, 12, 3.036),
    (1.0, 12, 4.048),
    (1.0, 6, 2.024),
    (1.0, 6, 3.036),
    (1.0, 6, 4.048),
    (1.0, 9, 2.024),
    (1.0, 9, 3.036),
    (1.0, 9, 4.048),
    (1.25, 9.9645, 3.036),
    (1.5, 10.929, 2.024),
    (1.5, 10.929, 3.036),
    (1.5, 10.929, 4.048),
    (1.5, 13.929, 2.024),
    (1.5, 13.929, 3.036),
    (1.5, 13.929, 4.048),
    (1.5, 7.929, 2.024),
    (1.5, 7.929, 3.036),
    (1.5, 7.929, 4.048),
]

# Generate geometries for all cases
for alpha, L, ceiling_param in cases:
    print(f"Generating geometry for alpha={alpha}, L={L}, ceiling_param={ceiling_param}")
    create_perhill_geom_ansys(alpha=alpha, L=L, ceiling_param=ceiling_param)

print("All geometries have been generated.")