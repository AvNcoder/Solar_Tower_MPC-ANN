import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# 1. RESEARCH-GRADE ANN ARCHITECTURE
# Matches the solar_tower_ann.pth structure (64 -> 32 -> 1)
class SolarMPCNet(nn.Module):
    def __init__(self, input_dim):
        super(SolarMPCNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

def calculate_thermal_power(flow, inlet_temp, outlet_temp_target=565.0):
    """Calculates thermal power in MW: P = (m_dot * Cp * delta_T) / 1000"""
    # Cp for molten salt ~ 1.51 kJ/kg-K
    return (flow * 1.51 * (outlet_temp_target - inlet_temp)) / 1000

def generate_research_plots():
    # 2. LOAD DATA
    # Ensure tower_mpc_dataset_v2.csv is in the current directory
    try:
        df = pd.read_csv('data/tower_mpc_dataset_v3.csv')
    except FileNotFoundError:
        print("Error: dataset file not found.")
        return

    # Map inputs for the 5-dim model
    input_cols = ['DNI_Irradiance', 'Inlet_Temp', 'Previous_Salt_Flow']
    target_col = 'Optimal_Flow_Target'
    
    X_raw = df[input_cols].values.astype(np.float32)
    y_raw = df[target_col].values.astype(np.float32).reshape(-1, 1)
    
    time_min = df['Time_Step_Min'].values
    dni = df['DNI_Irradiance'].values
    inlet_temp = df['Inlet_Temp'].values

    # 3. PREPROCESSING
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw)

    # 4. LOAD THE TRAINED MODEL
    model = SolarMPCNet(input_dim=3)
    model.load_state_dict(torch.load('models/solar_tower_ann.pth', map_location=torch.device('cpu')))
    model.eval()

    # 5. INFERENCE
    with torch.no_grad():
        preds_scaled = model(torch.from_numpy(X_scaled).float()).numpy()
    
    predictions = scaler_y.inverse_transform(preds_scaled).flatten()
    actuals = y_raw.flatten()

    # 6. CALCULATE SECONDARY METRICS
    power_actual = calculate_thermal_power(actuals, inlet_temp)
    power_pred = calculate_thermal_power(predictions, inlet_temp)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    errors = predictions - actuals

    # 7. VISUALIZATION - 5-PANEL RESEARCH FIGURE
    plt.style.use('seaborn-v0_8-paper')
    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)

    # PANEL A: Transient Flow Tracking
    ax0 = fig.add_subplot(gs[0, 0])
    subset = slice(0, 150) # Look at first 150 samples (12.5 hours)
    ax0.plot(time_min[subset], actuals[subset], label='Physics Model', color='#000000', linewidth=2.5, alpha=0.6)
    ax0.plot(time_min[subset], predictions[subset], label='ANN Surrogate', color='#d62728', linestyle='--', linewidth=1.5)
    ax0.set_title('A: Dynamic Response Tracking', fontsize=14, fontweight='bold')
    ax0.set_ylabel('Molten Salt Flow (kg/s)', fontsize=12)
    ax0.set_xlabel('Time (Minutes)', fontsize=12)
    ax0.legend()
    ax0.grid(True, linestyle=':', alpha=0.6)

    # PANEL B: Parity Plot
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.scatter(actuals, predictions, alpha=0.3, color='#2ca02c', s=10)
    ideal_line = [actuals.min(), actuals.max()]
    ax1.plot(ideal_line, ideal_line, 'k-', alpha=0.7, label='Ideal ($y=x$)')
    ax1.set_title('B: Prediction Correlation', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Measured Physics Flow (kg/s)', fontsize=12)
    ax1.set_ylabel('Predicted ANN Flow (kg/s)', fontsize=12)
    stats_text = f'$R^2 = {r2:.4f}$\n$RMSE = {rmse:.2f}$ kg/s'
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.grid(True, linestyle=':', alpha=0.6)

    # PANEL C: Error Distribution
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(errors, bins=40, color='#9467bd', edgecolor='white', alpha=0.8)
    ax2.axvline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_title('C: Residual Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Prediction Error (kg/s)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # PANEL D: Solar Irradiance (DNI) vs Time
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(time_min, dni, color='#ff7f0e', lw=2)
    ax3.set_title('D: Solar Irradiance (DNI) vs Time', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time (Minutes)', fontsize=12)
    ax3.set_ylabel('DNI ($W/m^2$)', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # PANEL E: Thermal Power Generation
    ax4 = fig.add_subplot(gs[1, 1:]) # Spans last two columns
    ax4.plot(time_min, power_actual, label='Physics Power', color='#000000', lw=2)
    ax4.plot(time_min, power_pred, label='ANN Pred Power', color='#d62728', linestyle=':', lw=1.5)
    ax4.set_title('E: Thermal Power Generation Comparison', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Time (Minutes)', fontsize=12)
    ax4.set_ylabel('Thermal Power (MW)', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Performance Verification of Solar Tower Control ANN', fontsize=20, y=1.0, fontweight='bold')
    plt.savefig('solar_research_final_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    generate_research_plots()