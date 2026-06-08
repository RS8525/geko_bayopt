import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    file_path = Path("data/dns/ffs/FFS_Reh6000_SBES_Node_2D.csv")
    
    if not file_path.exists():
        print(f"Error: {file_path} not found. Please run average_z_dns_ffs.py first.")
        return
        
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    
    # Extract coordinates and variables
    x = df['x-coordinate']
    y = df['y-coordinate']
    mean_k = df['mean-turbulent-kinetic-energy--k-dataset']
    mean_u = df['mean-x-velocity']
    
    print(f"Plotting {len(df)} points...")
    
    # Create a figure with 3 stacked subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # 1. Mesh (Nodes)
    axs[0].scatter(x, y, s=1, color='black', alpha=0.5, marker='.')
    axs[0].set_title('Mesh Nodes (Averaged 2D Grid)')
    axs[0].set_ylabel('y')
    axs[0].set_aspect('equal')
    
    # 2. Mean Turbulent Kinetic Energy
    sc_k = axs[1].scatter(x, y, c=mean_k, cmap='viridis', s=2, marker='.')
    axs[1].set_title('Mean Turbulent Kinetic Energy')
    axs[1].set_ylabel('y')
    axs[1].set_aspect('equal')
    fig.colorbar(sc_k, ax=axs[1], label='Turbulent Kinetic Energy')
    
    # 3. Mean X Velocity
    sc_u = axs[2].scatter(x, y, c=mean_u, cmap='coolwarm', s=2, marker='.')
    axs[2].set_title('Mean X-Velocity')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('y')
    axs[2].set_aspect('equal')
    fig.colorbar(sc_u, ax=axs[2], label='U Velocity')
    
    plt.tight_layout()
    
    # Save the figure and also show it
    out_file = Path("tests/ffs_dns_2d_plot.png")
    plt.savefig(out_file, dpi=200)
    print(f"Plot saved to {out_file}.")
    plt.show()

if __name__ == "__main__":
    main()
