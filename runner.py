"""Runner script for executing RZNE benchmarks.

This script runs various quantum circuit benchmarks and saves results to CSV files.
"""

import csv
import os
from datetime import date

import rzne

# Generate date string for output directory
today = date.today()
today_str = today.strftime("%Y-%m-%d")
def write_results_to_csv(filename, data_list):
    """Write results to CSV file.
    
    Args:
        filename: Path to CSV file
        data_list: List of data rows to write
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for data in data_list:
            if data:  # Only write non-empty results
                writer.writerow(data)


def main():
    """Main execution function."""
    dirname = f"Test_runs_decoh_depol_readout_paper_{today_str}"
    
    # Save noise model
    fname = os.path.join(dirname, "noisemodel.pkl")
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    rzne.save_object(rzne.noise_model, fname)
    
    # Run Hamiltonian Simulation benchmarks
    filename = os.path.join(dirname, "Hamsim.csv")
    dirname2 = os.path.join(dirname, "Hamsim_raw_files")
    hamsim_results = []
    for i in range(4, 8):
        for j in range(4, 12, 2):
            data = rzne.execute_hamsimul_test(i, j, dirname2)
            if data:
                hamsim_results.append(data)
    write_results_to_csv(filename, hamsim_results)
    
    # Run VQE benchmarks
    filename = os.path.join(dirname, "VQE.csv")
    dirname2 = os.path.join(dirname, "VQE_raw_files")
    vqe_results = []
    for i in range(4, 8):
        for j in range(4, 12, 2):
            data = rzne.execute_VQE(i, j, dirname2)
            if data:
                vqe_results.append(data)
    write_results_to_csv(filename, vqe_results)
    
    # Run QAOA benchmarks
    filename = os.path.join(dirname, "QAOA.csv")
    dirname2 = os.path.join(dirname, "QAOA_raw_files")
    qaoa_results = []
    for i in range(4, 8):
        for j in range(4, 12, 2):
            data = rzne.execute_QAOA(i, j, dirname2)
            if data:
                qaoa_results.append(data)
    write_results_to_csv(filename, qaoa_results)
    
    # Run QAOA Swap benchmarks
    filename = os.path.join(dirname, "QAOASwap.csv")
    dirname2 = os.path.join(dirname, "QAOASwap_raw_files")
    qaoa_swap_results = []
    for i in range(4, 8):
        for j in range(4, 12, 2):
            data = rzne.execute_QAOA_Swap(i, j, dirname2)
            if data:
                qaoa_swap_results.append(data)
    write_results_to_csv(filename, qaoa_swap_results)


if __name__ == "__main__":
    main()

    # Uncomment to run GHZ and MerminBell benchmarks
    # filename = os.path.join(dirname, "GHZ.csv")
    # dirname2 = os.path.join(dirname, "GHZ_raw_files")
    # for i in range(4, 17, 4):
    #     data = rzne.execute_GHZ(i, 0, dirname2)
    #     write_results_to_csv(filename, [data])
    #
    # filename = os.path.join(dirname, "MerminBell.csv")
    # dirname2 = os.path.join(dirname, "MerminBell_raw_files")
    # for i in range(4, 8):
    #     data = rzne.execute_MerminBell(i, 0, dirname2)
    #     write_results_to_csv(filename, [data])
