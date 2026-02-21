import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib
from src.neural_net import SolarMPCNet

def evaluate():
    # 1. Load Everything
    X_raw = np.load('data/inputs.npy')
    y_raw = np.load('data/targets.npy')
    scaler_x = joblib.load('models/scaler_x.pkl')
    scaler_y = joblib.load('models/scaler_y.pkl')
    
    model = SolarMPCNet(input_dim=5)
    model.load_state_dict(torch.load('models/solar_tower_ann.pth'))
    model.eval()

    # 2. Predict
    X_scaled = scaler_x.transform(X_raw)
    with torch.no_grad():
        preds_scaled = model(torch.from_numpy(X_scaled).float()).numpy()
    
    # 3. Denormalize
    predictions = scaler_y.inverse_transform(preds_scaled).flatten()
    actuals = y_raw.flatten()

    # 4. Calculate Research Metrics
    mse = np.mean((actuals - predictions)**2)
    aaci_actual = np.sum(np.abs(np.diff(actuals)))
    aaci_pred = np.sum(np.abs(np.diff(predictions)))

    print(f"\n" + "="*30)
    print("FINAL PERFORMANCE VERIFICATION")
    print("="*30)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Teacher Smoothness (AACI): {aaci_actual:.2f}")
    print(f"Student Smoothness (AACI): {aaci_pred:.2f}")
    print(f"Smoothness Ratio: {aaci_pred/aaci_actual:.4f} (Ideal: 1.0)")
    print("="*30)

    # 5. Plot Comparison
    plt.figure(figsize=(12, 5))
    plt.plot(actuals[:150], label='Teacher (Physics Model)', color='blue', alpha=0.6, linestyle='--')
    plt.plot(predictions[:150], label='Student (ANN)', color='red', alpha=0.8)
    plt.title('Solar Tower Flow Control: ANN vs Physics Teacher')
    plt.ylabel('Molten Salt Flow (kg/s)')
    plt.xlabel('Time Steps (5-min intervals)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('evaluation_results.png')
    print("Plot saved as 'evaluation_results.png'")

if __name__ == "__main__":
    evaluate()