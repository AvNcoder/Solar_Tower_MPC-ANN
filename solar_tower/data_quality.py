import numpy as np
import matplotlib.pyplot as plt
import os

def check_data_quality(target_file='data/targets.npy'):
    """
    Evaluates the 'Teacher' data to ensure it is smooth enough 
    for the Neural Network to learn stable control.
    """
    if not os.path.exists(target_file):
        print(f"Error: {target_file} not found. Run generate_dataset.py first.")
        return

    # 1. Load the target flow rates
    targets = np.load(target_file)
    
    # 2. Calculate AACI (Accumulated Absolute Control Increment)
    # This measures the sum of all 'jumps' the pump makes.
    # Lower = Smoother/Better for real hardware.
    diffs = np.abs(np.diff(targets.flatten()))
    aaci = np.sum(diffs)
    mean_jump = np.mean(diffs)

    print("--- Teacher Data Quality Report ---")
    print(f"Total Samples: {len(targets)}")
    print(f"Total Jitter (AACI): {aaci:.2f}")
    print(f"Average Jump per Step: {mean_jump:.2f} kg/s")

    # 3. Visual Inspection
    plt.figure(figsize=(12, 5))
    
    # Plotting first 300 steps (approx 25 hours of operation)
    plt.plot(targets[:300], label='Optimal Flow Target (Teacher)', color='#2ca02c', linewidth=1.5)
    
    plt.title(f'Teacher Action Smoothness (AACI: {aaci:.2f})', fontsize=14)
    plt.xlabel('Time Steps (5-min intervals)', fontsize=12)
    plt.ylabel('Flow Rate (kg/s)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Save the quality report image
    plt.savefig('data_quality_report.png')
    print("Report saved as 'data_quality_report.png'")
    plt.show()

if __name__ == "__main__":
    # Ensure the path matches where your generate_dataset.py saves files
    check_data_quality('data/targets.npy')