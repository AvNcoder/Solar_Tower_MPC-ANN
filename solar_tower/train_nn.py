import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import joblib  # To save the scaler for later use in evaluation
from sklearn.preprocessing import StandardScaler
from src.neural_net import SolarMPCNet

def train_model():
    # 1. Setup Paths
    data_path = 'data/inputs.npy'
    target_path = 'data/targets.npy'
    model_dir = 'models'
    
    if not os.path.exists(data_path):
        print("Data files not found. Please run the generation script first.")
        return

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 2. Load Data
    X_raw = np.load(data_path).astype(np.float32)
    y_raw = np.load(target_path).astype(np.float32)

    # 3. Scaling (Standardization: Mean=0, Unit Variance)
    # Research show StandardScaling helps ANN converge faster for thermal systems
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_x.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1))

    # Save scalers to use during prediction/inference later
    joblib.dump(scaler_x, f'{model_dir}/scaler_x.pkl')
    joblib.dump(scaler_y, f'{model_dir}/scaler_y.pkl')

    # Convert to Tensors
    inputs = torch.from_numpy(X_scaled)
    targets = torch.from_numpy(y_scaled)

    # 4. Initialize Network
    # input_dim should be 5: Elevation, Azimuth, DNI, Inlet_Temp, Prev_Flow
    model = SolarMPCNet(input_dim=X_raw.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Lower LR for stability

    # 5. Training Loop
    print(f"--- Starting Training on {X_raw.shape[0]} samples ---")
    epochs = 1000 
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    # 6. Save Model
    torch.save(model.state_dict(), f'{model_dir}/solar_tower_ann.pth')
    print(f"--- Training Complete ---")
    print(f"Model saved to {model_dir}/solar_tower_ann.pth")
    print(f"Scalers saved to {model_dir}/scaler_x.pkl and scaler_y.pkl")

if __name__ == "__main__":
    train_model()