import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

# --- 1. DNI DISTURBANCE LOGIC ---
def apply_cloud_disturbances(dni_clear, cloud_probability=0.2, severity=0.6):
    """
    Introduces stochastic cloud transients to the smooth DNI.
    - cloud_probability: Chance of a cloud event starting.
    - severity: How much of the sun is blocked (0.1 = 90% drop).
    """
    np.random.seed(42)
    disturbed_dni = np.copy(dni_clear)
    
    i = 0
    while i < len(disturbed_dni):
        if disturbed_dni[i] > 0 and np.random.random() < cloud_probability:
            # Cloud duration: 3 to 15 time steps (15-75 mins)
            duration = np.random.randint(3, 15)
            # Apply a dip in irradiance
            disturbed_dni[i:i+duration] *= np.random.uniform(severity, 0.9)
            i += duration
        else:
            i += 1
    return disturbed_dni



   #  --- HEAT LOSS---
   
def calculate_heat_losses(t_receiver, t_ambient=25.0):
    """
    Calculates environmental losses (Radiation + Convection).
    Based on Stefan-Boltzmann Law and Newton's Law of Cooling.
    """
    h_conv = 0.015   # Convective coefficient (kW/m2-K)
    sigma = 5.67e-11 # Stefan-Boltzmann constant (kW/m2-K4)
    epsilon = 0.88   # Emissivity from research paper
    area = 550       # Receiver Area (m2)
    
    # Radiation Loss
    q_rad = epsilon * sigma * area * ((t_receiver + 273.15)**4 - (t_ambient + 273.15)**4)
    # Convection Loss
    q_conv = h_conv * area * (t_receiver - t_ambient)
    
    return q_rad + q_conv

# --- 2. DATA GENERATION LOGIC ---

def generate_spt_dataset(num_samples=1000, timestep_min=5, include_clouds=True):
    """Generates dataset using Dynamic Energy Balance ODEs from research paper."""
    # Physical Constants from Paper [cite: 110, 143]
    cp_salt = 1.51       # kJ/kg-K (Molten salt specific heat)
    cp_tube = 0.499      # kJ/kg-K (Tube wall specific heat )
    m_tube_total = 5000  # kg (Estimated thermal mass of receiver tubes)
    target_tout = 565.0  # Target [cite: 86]
    tin_nominal = 290.0  # Inlet [cite: 86]
    receiver_area = 550  
    eta_receiver = 0.88  # Emissivity [cite: 127]
    dt_sec = timestep_min * 60 # Convert 5-min to 300s for ODE integration
    
    # A. Predictable Clear-Sky DNI (Sine Wave)
    steps_per_day = 288
    time_array = np.arange(num_samples)
    dni_sine = 1000 * np.sin(2 * np.pi * time_array / steps_per_day - (np.pi/2))
    dni_clear = np.maximum(0, dni_sine) 

    # B. Apply Disturbances
    dni_final = apply_cloud_disturbances(dni_clear) if include_clouds else dni_clear

    # C. Dynamic ODE Loop (Conservation of Energy) 
    tin = np.random.normal(tin_nominal, 2, num_samples)
    tout_dynamic = np.zeros(num_samples)
    optimal_flow = np.zeros(num_samples)
    
    # Initialize state (Start at inlet temp)
    current_tout = tin[0] 

    for t in range(num_samples):
        # 1. Energy In from Sun
        q_in = receiver_area * dni_final[t] * eta_receiver 
        
        # 2. Calculate Environmental Heat Loss (Radiation + Convection)
        q_loss = calculate_heat_losses(current_tout)
        
        # 3. Solve for 'Optimal Flow' using Net Power
        # The salt only receives power that isn't lost to the atmosphere
        if dni_final[t] > 50: 
            m_dot_target = (q_in - q_loss) / (cp_salt * (target_tout - tin[t]))
        else:
            m_dot_target = 0
            
        # 4. Dynamic ODE (Conservation of Energy)
        # Net Change = Solar Gain - Energy removed by Salt - Heat Loss
        q_salt = m_dot_target * cp_salt * (current_tout - tin[t])
        dT = (dt_sec / (m_tube_total * cp_tube)) * (q_in - q_salt - q_loss)
        
        # 5. Update State & Apply Safety Constraints
        current_tout += dT
        current_tout = max(tin[t], min(current_tout, 600)) 
        
        tout_dynamic[t] = current_tout
        optimal_flow[t] = max(0, m_dot_target)

    # D. Feature Engineering
    prev_flow = np.roll(optimal_flow, 1)
    prev_flow[0] = 0.0 

    df = pd.DataFrame({
        'Time_Step_Min': time_array * timestep_min,
        'DNI_Irradiance': dni_final.round(2),
        'Inlet_Temp': tin.round(2),
        'Current_Outlet_Temp': tout_dynamic.round(2), # Added dynamic tracking
        'Previous_Salt_Flow': prev_flow.round(4), 
        'Optimal_Flow_Target': optimal_flow.round(4)
    })
    return df


# --- 2. EXPORT LOGIC ---
def export_to_numpy(df, input_file='data/inputs.npy', target_file='data/targets.npy'):
    input_cols = ['DNI_Irradiance', 'Inlet_Temp', 'Previous_Salt_Flow']
    target_col = ['Optimal_Flow_Target']
    
    inputs_array = df[input_cols].values
    targets_array = df[target_col].values
    
    np.save(input_file, inputs_array)
    np.save(target_file, targets_array)
    print(f"NumPy export complete: {inputs_array.shape}")


# --- 3. VISUALIZATION AUDIT ---
def plot_dni_profile(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df['Time_Step_Min'][:500], df['DNI_Irradiance'][:500], color='#ff7f0e', label='DNI with Clouds')
    plt.fill_between(df['Time_Step_Min'][:500], 0, df['DNI_Irradiance'][:500], color='#ff7f0e', alpha=0.1)
    plt.title('Predictable DNI Profile with Cloud Disturbances')
    plt.ylabel('Irradiance (W/m²)')
    plt.xlabel('Time (Minutes)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig('dni_disturbance_profile.png')
    print("DNI profile saved to 'dni_disturbance_profile.png'")

if __name__ == "__main__":
    if not os.path.exists('data'): os.makedirs('data')

    # Generate Research Dataset
    df = generate_spt_dataset(num_samples=1500, include_clouds=True)
    
    # Export
    df.to_csv('data/tower_mpc_dataset_v3.csv', index=False)
    export_to_numpy(df)
    
    
    # Audit
    plot_dni_profile(df)
    print("Head of dataset:")
    print(df.head(10))
    
    
    