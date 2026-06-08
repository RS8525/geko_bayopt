import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_results():
    csv_path = Path(r"results\experiments\periodic_hills_csep_pmdb_v2\metadata.csv")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Skip first row in csv (first data row) as requested
    df = df.iloc[1:]

    # Extract required columns (X-Axis: C_SEP, Y-Axis: Score)
    x = df["geko_csep"]
    y = df["score"]
    
    # Create the scatter plot. Using the index as the colorbar to indicate iteration order.
    # If a specific 'collum 5' was intended other than geko_csep, it can be swapped below.
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x, y, c=df.index, cmap="viridis", s=50, alpha=0.8, edgecolors="black")
    
    plt.title("BO Sweep: Score vs C_SEP")
    plt.xlabel("C_SEP")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Iteration (Row Index)")
    
    # Save the figure in the same folder as metadata.csv
    save_path = csv_path.parent / "metadata_plot.png"
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    plot_results()