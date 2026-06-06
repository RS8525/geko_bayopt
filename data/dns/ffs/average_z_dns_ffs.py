import pandas as pd
import numpy as np
import time
from pathlib import Path

def main():
    input_file = Path("data/dns/ffs/FFS_Reh6000_SBES_Node")
    output_file = Path("data/dns/ffs/FFS_Reh6000_SBES_Node_2D.csv")
    
    print(f"Reading {input_file} ... (this might take a minute)")
    start_time = time.time()
    
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip()
    
    print(f"Loaded {len(df)} rows in {time.time() - start_time:.2f} seconds.")
    
    print("Integrating over z-coordinate and averaging over z-length...")
    
    # --- z-integration weights: trapezoidal rule ---
    z_vals = np.sort(df["z-coordinate"].unique())
    
    if len(z_vals) < 2:
        raise ValueError("Need at least two z-planes to integrate over z.")
    
    z_weights = np.zeros_like(z_vals, dtype=float)
    z_weights[0] = 0.5 * (z_vals[1] - z_vals[0])
    z_weights[-1] = 0.5 * (z_vals[-1] - z_vals[-2])
    
    if len(z_vals) > 2:
        z_weights[1:-1] = 0.5 * (z_vals[2:] - z_vals[:-2])
    
    z_weight_map = dict(zip(z_vals, z_weights))
    df["_z_weight"] = df["z-coordinate"].map(z_weight_map)
    
    # Columns not to average as physical fields
    non_field_cols = {
        "nodenumber",
        "x-coordinate",
        "y-coordinate",
        "z-coordinate",
        "_z_weight",
    }
    
    field_cols = [
        c for c in df.columns
        if c not in non_field_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    
    # Integrate: phi * dz
    df[field_cols] = df[field_cols].mul(df["_z_weight"], axis=0)
    df["_z_length"] = df["_z_weight"]
    
    # Sum integral at each (x,y)
    df_mean = (
        df[["x-coordinate", "y-coordinate"] + field_cols + ["_z_length"]]
        .groupby(["x-coordinate", "y-coordinate"], as_index=False)
        .sum()
    )
    
    # Average over z-length: integral / length
    df_mean[field_cols] = df_mean[field_cols].div(df_mean["_z_length"], axis=0)
    df_mean = df_mean.drop(columns=["_z_length"])
    
    print(f"Reduced to {len(df_mean)} unique 2D points.")
    
    print(f"Saving to {output_file} ...")
    df_mean.to_csv(output_file, index=False)
    
    print(f"Done! Total time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()