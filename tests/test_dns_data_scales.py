import os
import numpy as np

def analyze_dns_data(case_name):
    """
    Analyzes DNS data to find specific maxima and checks constraints.

    Args:
        case_name (str): The name of the DNS case directory.
    """
    # Define the path to the data file
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "dns", "periodic_hills", "pehill-29-cases-DNS"))
    data_file = os.path.join(base_dir, case_name, "mean_files.dat")

    # Load the data
    data = np.genfromtxt(data_file, dtype=float, skip_header=0, delimiter=None)

    # Find max in column 2 and its location (x, y)
    col_2 = data[:, 2]
    max_col_2_index = np.argmax(col_2)
    max_col_2_value = col_2[max_col_2_index]
    max_col_2_location = data[max_col_2_index, :2]  # First two columns for (x, y)

    print(f"Max in column 2: {max_col_2_value} at location (x, y): {max_col_2_location}")

    # Find max in column 6 under the constraint of max in column 0
    col_0 = data[:, 0]
    max_col_0_value = np.max(col_0)
    constrained_data_max = data[data[:, 0] == max_col_0_value]  # Filter rows where column 0 is max
    constrained_data_min = data[data[:, 0] == np.min(col_0)]  # Filter rows where column 0 is min
    col_5 = constrained_data_max[:, 5]  # Column with index 5 (6th column)
    col_2 = constrained_data_min[:, 2]  # Column with index 2 (3rd column)
    max_col_5_value = np.max(col_5)
    min_col_2_value = np.min(col_2)


    print(f"Max in column 5 under constraint of max in column 0: {max_col_5_value}")
    print(f"Min in column 5 under constraint of max in column 0: {np.min(col_5)}")
    print(f"Max in column 2 under constraint of min in column 0: {np.max(col_2)}")
    print(f"Min in column 2 under constraint of min in column 0: {min_col_2_value}")
    

    # Check if column 5 attains 1, 2, or more different values for max in column 0
    col_5_values = constrained_data_max[:, 5]
    unique_col_5_values = np.unique(col_5_values)

    if len(unique_col_5_values) == 1:
        print("Column 5 attains 1 unique value for max in column 0.")
    elif len(unique_col_5_values) == 2:
        print("Column 5 attains 2 unique values for max in column 0.")
    else:
        print(f"Column 5 attains {len(unique_col_5_values)} unique values for max in column 0.")
    
    print(f"Unique values in column 2 for min in column 0: {len(np.unique(col_2))}")


if __name__ == "__main__":
    # Example usage
    analyze_dns_data("alph10-9-3036")